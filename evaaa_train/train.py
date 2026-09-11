import os, importlib, warnings, shutil
from typing import Any, Dict
import sys

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.distributions import Distribution

from lightning import Fabric

from cli import resume_from_checkpoint
from utils.metric import MetricAggregator
from utils.timer import timer
from utils.utils import dotdict, print_config

def is_dqn_selected():
    for arg in sys.argv:
        if "exp=dqn" in arg:
            return True
    return False

if is_dqn_selected():
    import algos.dqn.dqn
    algos.dqn.dqn.main()
    sys.exit(0)

@hydra.main(version_base="1.3", config_path="configs", config_name="config")
def main(cfg: DictConfig):
    print_config(cfg)

    cfg = dotdict(OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True))

    # Torch settings
    os.environ["OMP_NUM_THREADS"] = str(cfg.num_threads)
    torch.set_float32_matmul_precision(cfg.float32_matmul_precision)

    Distribution.set_default_validate_args(cfg.distribution.validate_args)

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
    
    # Save the environment configuration
    cfg_save_dir = HydraConfig.get().runtime.output_dir
    env_cfg_save_dir = os.path.join(cfg_save_dir, "env_cfg")
    os.makedirs(env_cfg_save_dir, exist_ok=True)

    env_cfg = cfg.env.wrapper.env_cfg
    env_cfg_dir =  os.path.join(cfg.env.base_dir, env_cfg["env"]["env_name"], "Config", cfg.env.config)

    dst_dir = os.path.join(env_cfg_save_dir, os.path.basename(env_cfg_dir))

    shutil.copytree(env_cfg_dir, dst_dir, dirs_exist_ok=True)

    # Launch the algorithm
    fabric.launch(reproducible(command), cfg, **kwargs)

if __name__ == "__main__":
    main()