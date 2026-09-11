import os, sys
import hydra
from hydra.core.hydra_config import HydraConfig
import importlib
from pathlib import Path

from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from torch.distributions import Distribution

from lightning import Fabric

from cli import check_configs_evaluation
from utils.utils import dotdict

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
   # Load the checkpoint configuration
    checkpoint_path = Path(os.path.abspath(cfg.checkpoint_path))

    # Merge the two configs
    with open_dict(cfg):
        cfg.root_dir = str(checkpoint_path.parent.parent.parent.parent)

        ckpt_cfg = cfg

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

def check_configs_evaluation(cfg: DictConfig):
    if cfg.float32_matmul_precision not in {"medium", "high", "highest"}:
        raise ValueError(
            f"Invalid value '{cfg.float32_matmul_precision}' for the 'float32_matmul_precision' parameter. "
            "It must be one of 'medium', 'high' or 'highest'."
        )
    if cfg.checkpoint_path is None:
        raise ValueError("You must specify the evaluation checkpoint path")

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

    module = f"algos.{cfg.algo.name}"
    # module = "algos.dreamer_v3"
    # module = "algos.ppo"
    evaluation_file = "evaluate"
    entrypoint = "evaluate"

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

if __name__ == "__main__":
    main()
