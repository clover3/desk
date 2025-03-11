from rule_gen.reddit.classifier_loader.inst_builder import get_instruction_from_run_name, get_no_rule_instruction
from rule_gen.reddit.path_helper import get_split_subreddit_list

def get_prompt_factory_both_rule():
    sb_list = get_split_subreddit_list("train")
    instruction_d = {}
    for sb in sb_list:
        try:
            run_name = f"api_{sb}_both"
            instruction, _ = get_instruction_from_run_name(run_name)
        except FileNotFoundError:
            instruction, _ = get_no_rule_instruction(run_name)
        instruction_d[sb] = instruction

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt



def get_prompt_factory_no_rule():
    sb_list = get_split_subreddit_list("train")
    instruction_d = {}
    for sb in sb_list:
        run_name = f"api_{sb}_both"
        instruction, _ = get_no_rule_instruction(run_name)
        instruction_d[sb] = instruction

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt


def get_prompt_fn_from_type(prompt_type):
    if prompt_type == "both_rule":
        get_prompt_fn = get_prompt_factory_both_rule()
    elif prompt_type == "sb_name":
        get_prompt_fn = get_prompt_factory_no_rule()
    else:
        raise ValueError()
    return get_prompt_fn

