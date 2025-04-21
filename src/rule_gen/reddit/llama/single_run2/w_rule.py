from rule_gen.reddit.llama.prompt_helper import get_prompt_fn_from_type

from rule_gen.reddit.llama.single_run.make_json_payload import save_json_payload


def main3():
    src_data_name = "train_comb5"
    prompt_type = "both_rule"
    get_prompt = get_prompt_fn_from_type(prompt_type)

    save_data_name = "both_rule_comb5"
    root_data_name = "train_data3"

    for role in ["train", "val"]:
        save_json_payload(
            root_data_name,
            src_data_name,
            save_data_name,
            get_prompt, role)


if __name__ == "__main__":
    main3()
