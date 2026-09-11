import importlib
import os
import pathlib
import warnings
from pathlib import Path
from typing import Any, Dict

import hydra
import torch
from lightning import Fabric
from lightning.fabric.strategies import STRATEGY_REGISTRY, DDPStrategy, SingleDeviceStrategy, Strategy
from omegaconf import DictConfig, OmegaConf, open_dict
from torch.distributions import Distribution

from utils.imports import _IS_MLFLOW_AVAILABLE
from utils.logger import get_logger
from utils.metric import MetricAggregator
from utils.registry import algorithm_registry, evaluation_registry
from utils.timer import timer
from utils.utils import dotdict, print_config

def resume_from_checkpoint(cfg: DictConfig) -> DictConfig:
    ckpt_path = pathlib.Path(cfg.checkpoint.resume_from)
    old_cfg = OmegaConf.load(ckpt_path.parent.parent / "config.yaml")
    old_cfg = dotdict(OmegaConf.to_container(old_cfg, resolve=True, throw_on_missing=True))
    if old_cfg.env.id != cfg.env.id:
        raise ValueError(
            "This experiment is run with a different environment from the one of the experiment you want to restart. "
            f"Got '{cfg.env.id}', but the environment of the experiment of the checkpoint was {old_cfg.env.id}. "
            "Set properly the environment for restarting the experiment."
        )
    if old_cfg.algo.name != cfg.algo.name:
        raise ValueError(
            "This experiment is run with a different algorithm from the one of the experiment you want to restart. "
            f"Got '{cfg.algo.name}', but the algorithm of the experiment of the checkpoint was {old_cfg.algo.name}. "
            "Set properly the algorithm name for restarting the experiment."
        )
    if old_cfg.algo.learning_starts > 0:
        warnings.warn(
            "The `algo.learning_starts` parameter is greater than zero. "
            "This means that the resuming experiment will pre-fill the buffer for `algo.learning_starts` steps. "
            "If this is not intended please set the `algo.learning_starts=0` parameter in the experiment configuration "
            "or through the CLI."
        )

    # Remove keys from the `old_cfg` that must not be overridden
    old_cfg.pop("root_dir", None)
    old_cfg.pop("run_name", None)
    old_cfg.algo.pop("total_steps", None)
    old_cfg.algo.pop("learning_starts", None)
    old_cfg.checkpoint.pop("resume_from", None)
    # Substitute the config with the old one (except for the parameters removed before)
    # because the experiment must continue with the same parameters
    with open_dict(cfg):
        cfg.merge_with(old_cfg.as_dict())
    return cfg

def run_algorithm(cfg: Dict[str, Any]):
    """Run the algorithm specified in the configuration.

    Args:
        cfg (Dict[str, Any]): the loaded configuration.
    """

    # Torch settings
    os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Set the distribution validate_args once here
    Distribution.set_default_validate_args(cfg.distribution.validate_args)

    # Given the algorithm's name, retrieve the module where
    # 'cfg.algo.name'.py is contained; from there retrieve the
    # 'register_algorithm'-decorated entrypoint;
    # the entrypoint will be launched by Fabric with 'fabric.launch(entrypoint)'
    module = None
    decoupled = False
    entrypoint = None
    algo_name = cfg.algo.name

    module = f"algos.{algo_name}"
    entrypoint = 'main'

    task = importlib.import_module(f"{module}.{algo_name}")
    utils = importlib.import_module(f"{module}.utils")
    command = task.__dict__[entrypoint]
    kwargs = {}

    strategy = cfg.fabric.get("strategy", "auto")

    fabric: Fabric = hydra.utils.instantiate(cfg.fabric, strategy=strategy, _convert_="all")

    if hasattr(cfg, "metric") and cfg.metric is not None:
        predefined_metric_keys = set()
        if not hasattr(utils, "AGGREGATOR_KEYS"):
            warnings.warn(
                f"No 'AGGREGATOR_KEYS' set found for the {algo_name} algorithm under the {module} module. "
                "No metric will be logged.",
                UserWarning,
            )
        else:
            predefined_metric_keys = utils.AGGREGATOR_KEYS
        timer.disabled = cfg.metric.log_level == 0 or cfg.metric.disable_timer
        keys_to_remove = set(cfg.metric.aggregator.metrics.keys()) - predefined_metric_keys
        for k in keys_to_remove:
            cfg.metric.aggregator.metrics.pop(k, None)
        MetricAggregator.disabled = cfg.metric.log_level == 0 or len(cfg.metric.aggregator.metrics) == 0

    # This function is used to make the algorithm reproducible.
    # It can be an overkill since Fabric already captures everything we're setting here
    # when multiprocessing is used with a `spawn` method (default with DDP strategy).
    # https://github.com/Lightning-AI/pytorch-lightning/blob/f23b3b1e7fdab1d325f79f69a28706d33144f27e/src/lightning/fabric/strategies/launchers/multiprocessing.py#L112
    def reproducible(func):
        def wrapper(fabric: Fabric, cfg: Dict[str, Any], *args, **kwargs):
            if cfg.cublas_workspace_config is not None:
                os.environ["CUBLAS_WORKSPACE_CONFIG"] = cfg.cublas_workspace_config
            fabric.seed_everything(cfg.seed)
            torch.backends.cudnn.benchmark = cfg.torch_backends_cudnn_benchmark
            torch.backends.cudnn.deterministic = cfg.torch_backends_cudnn_deterministic
            torch.use_deterministic_algorithms(cfg.torch_use_deterministic_algorithms)
            return func(fabric, cfg, *args, **kwargs)

        return wrapper

    fabric.launch(reproducible(command), cfg, **kwargs)


def eval_algorithm(cfg: DictConfig):
    """Run the algorithm specified in the configuration.

    Args:
        cfg (DictConfig): the loaded configuration.
    """
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # Torch settings
    os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    # Set the distribution validate_args once here
    Distribution.set_default_validate_args(cfg.distribution.validate_args)

    # TODO: change the number of devices when FSDP will be supported
    accelerator = cfg.fabric.get("accelerator", "auto")
    fabric: Fabric = hydra.utils.instantiate(
        cfg.fabric, accelerator=accelerator, devices=1, num_nodes=1, _convert_="all"
    )

    # Seed everything
    fabric.seed_everything(cfg.seed)

    # Load the checkpoint
    state = fabric.load(cfg.checkpoint_path)

    # Given the algorithm's name, retrieve the module where
    # 'cfg.algo.name'.py is contained; from there retrieve the
    # `register_algorithm`-decorated entrypoint;
    # the entrypoint will be launched by Fabric with `fabric.launch(entrypoint)`
    module = None
    entrypoint = None
    evaluation_file = None
    algo_name = cfg.algo.name
    for _module, _algos in evaluation_registry.items():
        for _algo in _algos:
            if algo_name == _algo["name"]:
                module = _module
                entrypoint = _algo["entrypoint"]
                evaluation_file = _algo["evaluation_file"]
                break
    if module is None:
        raise RuntimeError(f"Given the algorithm named `{algo_name}`, no module has been found to be imported.")
    if entrypoint is None:
        raise RuntimeError(
            f"Given the module and algorithm named `{module}` and `{algo_name}` respectively, "
            "no entrypoint has been found to be imported."
        )
    if evaluation_file is None:
        raise RuntimeError(
            f"Given the module and algorithm named `{module}` and `{algo_name}` respectively, "
            "no evaluation file has been found to be imported."
        )
    task = importlib.import_module(f"{module}.{evaluation_file}")
    command = task.__dict__[entrypoint]
    if getattr(cfg, "disable_grads", True):

        def no_grad(func):
            def wrapper(*args, **kwargs):
                with torch.no_grad():
                    return func(*args, **kwargs)

            return wrapper

        command = no_grad(command)
    fabric.launch(command, cfg, state)

def check_configs_evaluation(cfg: DictConfig):
    if cfg.float32_matmul_precision not in {"medium", "high", "highest"}:
        raise ValueError(
            f"Invalid value '{cfg.float32_matmul_precision}' for the 'float32_matmul_precision' parameter. "
            "It must be one of 'medium', 'high' or 'highest'."
        )
    if cfg.checkpoint_path is None:
        raise ValueError("You must specify the evaluation checkpoint path")


@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def run(cfg: DictConfig):
    """SheepRL zero-code command line utility."""
    print_config(cfg)
    if cfg.checkpoint.resume_from:
        cfg = resume_from_checkpoint(cfg)
    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
    # check_configs(cfg)
    run_algorithm(cfg)


@hydra.main(version_base="1.3", config_path="configs", config_name="eval_config")
def evaluation(cfg: DictConfig):
    # Load the checkpoint configuration
    checkpoint_path = Path(os.path.abspath(cfg.checkpoint_path))
    ckpt_cfg = OmegaConf.load(checkpoint_path.parent.parent / "config.yaml")
    ckpt_cfg.pop("seed", None)

    # Merge the two configs
    with open_dict(cfg):
        capture_video = getattr(cfg.env, "capture_video", True)
        cfg.env = {"capture_video": capture_video, "num_envs": 1}
        cfg.exp = {}
        cfg.algo = {}
        cfg.fabric = {
            "devices": 1,
            "num_nodes": 1,
            "strategy": "auto",
            "accelerator": getattr(cfg.fabric, "accelerator", "auto"),
        }
        cfg.root_dir = str(checkpoint_path.parent.parent.parent.parent)

        # Merge configs
        ckpt_cfg.merge_with(cfg)

        # Update values after merge
        run_name = Path(
            os.path.join(
                os.path.basename(checkpoint_path.parent.parent.parent),
                os.path.basename(checkpoint_path.parent.parent),
                "evaluation",
            )
        )
        ckpt_cfg.run_name = str(run_name)

    # Check the validity of the configuration and run the evaluation
    check_configs_evaluation(ckpt_cfg)
    eval_algorithm(ckpt_cfg)


# @hydra.main(version_base="1.3", config_path="configs", config_name="model_manager_config")
# def registration(cfg: DictConfig):
#     from sheeprl.utils.mlflow import register_model_from_checkpoint

#     checkpoint_path = Path(cfg.checkpoint_path)
#     ckpt_cfg = OmegaConf.load(checkpoint_path.parent.parent / "config.yaml")

#     # Merge the two configs
#     with open_dict(cfg):
#         cfg.env = ckpt_cfg.env
#         cfg.exp_name = ckpt_cfg.exp_name
#         cfg.algo = ckpt_cfg.algo
#         cfg.distribution = ckpt_cfg.distribution
#         cfg.seed = ckpt_cfg.seed

#     cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))
#     cfg.to_log = dotdict(OmegaConf.to_container(ckpt_cfg, resolve=True, throw_on_missing=True))

#     precision = getattr(ckpt_cfg.fabric, "precision", None)
#     fabric = Fabric(devices=1, accelerator="cpu", num_nodes=1, precision=precision)

#     # Load the checkpoint
#     state = fabric.load(cfg.checkpoint_path)
#     # Retrieve the algorithm name, used to import the custom
#     # log_models_from_checkpoint function.
#     algo_name = cfg.algo.name
#     if "decoupled" in cfg.algo.name:
#         algo_name = algo_name.replace("_decoupled", "")
#     if algo_name.startswith("p2e_dv"):
#         algo_name = "_".join(algo_name.split("_")[:2])
#     try:
#         log_models_from_checkpoint = importlib.import_module(
#             f"sheeprl.algos.{algo_name}.utils"
#         ).log_models_from_checkpoint
#     except Exception as e:
#         print(e)
#         raise RuntimeError(
#             f"Make sure that the algorithm is defined in the `./sheeprl/algos/{algo_name}` folder "
#             "and that the `log_models_from_checkpoint` function is defined "
#             f"in the `./sheeprl/algos/{algo_name}/utils.py` file."
#         )

#     fabric.launch(register_model_from_checkpoint, cfg, state, log_models_from_checkpoint)
