import os

from omegaconf import OmegaConf
import fire

from rule_gen.reddit.llama.lf_util import get_lf_config_path


def main(run_name):
    conf_path = os.path.join("confs", "lf", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    base_name = "sb_name"
    base_conf = OmegaConf.load(get_lf_config_path(base_name))

    new_conf = base_conf.copy()
    new_conf.dataset = conf.data_name
    new_conf.output_dir = conf.model_path
    OmegaConf.save(new_conf, get_lf_config_path(run_name))



if __name__ == "__main__":
    fire.Fire(main)
