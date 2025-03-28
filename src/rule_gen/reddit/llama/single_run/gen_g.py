from rule_gen.reddit.llama.common_lf import make_register_reddit_prompts_for_lf
from rule_gen.reddit.llama.prompt_helper import get_pattern_g_prompt_fn
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def main():
    src_data_name = "train_comb4"
    save_data_name = "lf_pattern_g"
    src_data_path = get_reddit_train_data_path_ex("train_data2", src_data_name, "train")
    get_prompt = get_pattern_g_prompt_fn()

    make_register_reddit_prompts_for_lf(get_prompt, save_data_name, src_data_path)


if __name__ == "__main__":
    main()
