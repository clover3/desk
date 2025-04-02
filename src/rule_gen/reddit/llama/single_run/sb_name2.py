from rule_gen.reddit.llama.prompt_helper import get_prompt_fn_from_type
from rule_gen.reddit.llama.single_run.make_json_payload import save_json_payload


def main():
    src_data_name = "train_comb4"
    prompt_type = "both_rule"
    get_prompt = get_prompt_fn_from_type(prompt_type)
    save_data_name = "both_rule_comb4"
    save_json_payload(src_data_name, save_data_name, get_prompt)


def main2():
    src_data_name = "train_comb4"
    prompt_type = "sb_name"
    get_prompt = get_prompt_fn_from_type(prompt_type)
    save_data_name = "sb_name_comb4"
    save_json_payload(src_data_name, save_data_name, get_prompt)


def main3():
    src_data_name = "train_comb4"
    prompt_type = "sb_name2"
    get_prompt = get_prompt_fn_from_type(prompt_type)
    save_data_name = "sb_name3"
    save_json_payload(src_data_name, save_data_name, get_prompt, "val")



if __name__ == "__main__":
    main3()
