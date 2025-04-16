from rule_gen.reddit.llama.prompt_helper import get_prompt_factory_no_rule2
from rule_gen.reddit.llama.single_run.make_json_payload import save_json_payload_from_train_data2, save_json_payload


def main3():
    src_data_name = "train_comb5"
    get_prompt = get_prompt_factory_no_rule2(1000)
    save_data_name = "sb_name5"
    root_data_name = "train_data3"

    for role in ["train", "val"]:
        save_json_payload(
            root_data_name,
            src_data_name,
            save_data_name,
            get_prompt, role)



if __name__ == "__main__":
    main3()
