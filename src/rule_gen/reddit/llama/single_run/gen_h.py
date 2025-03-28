import os

from omegaconf import OmegaConf

from rule_gen.reddit.llama.common_lf import make_register_reddit_prompts_for_lf
from rule_gen.reddit.llama.prompt_helper import get_prompt_fn_from_type
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def main():
    run_name = "lf_pattern8"
    conf_path = os.path.join("confs", "lf", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)

    src_data_name = "train_comb4"
    src_data_path = get_reddit_train_data_path_ex("train_data2", src_data_name, "train")
    get_prompt = get_prompt_fn_from_type(conf.prompt_type)
    make_register_reddit_prompts_for_lf(get_prompt, conf.data_name, src_data_path)


if __name__ == "__main__":
    main()
