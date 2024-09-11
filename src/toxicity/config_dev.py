import logging
import os

import hydra
from omegaconf import DictConfig, OmegaConf

import wandb
OmegaConf.register_new_resolver("uuid", lambda: 1)
LOG = logging.getLogger(__name__)


@hydra.main(config_path="config", config_name="config", version_base="1.2")
def run(config: DictConfig):
    if config.wandb:
        wandb.init(config={})
        if not config.wandb_run_name:
            try:
                wandb.run.name = f"{str(config.editor._name)}_{os.getenv('SLURM_JOBID')}"
            except:
                pass
        wandb.config = config
        wandb.log({"metric": 0.5})

    print(config)





if __name__ == "__main__":
    run()
