from rule_gen.reddit.llama.prompt_helper import get_prompt_factory_no_rule2
from rule_gen.reddit.llama.single_run.make_json_payload import save_json_payload_from_train_data2


def main3():
    src_data_name = "train_comb4"
    get_prompt = get_prompt_factory_no_rule2(1000)
    save_data_name = "sb_name4"
    save_json_payload_from_train_data2(src_data_name, save_data_name, get_prompt)
    save_json_payload_from_train_data2(src_data_name, save_data_name, get_prompt, "val")




if __name__ == "__main__":
    main3()
