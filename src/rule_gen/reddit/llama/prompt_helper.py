import json
import os
from json import JSONDecodeError

from rule_gen.cpath import output_root_path
from rule_gen.reddit.classifier_loader.inst_builder import get_instruction_from_run_name, get_no_rule_instruction
from rule_gen.reddit.llama.train_prompt_gen import get_pattern_instruction, get_pattern_instruction2, \
    get_pattern_instruction_w_prepost
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_subreddit_list


def get_prompt_factory_both_rule():
    sb_list = load_subreddit_list()
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
    sb_list = load_subreddit_list()
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
    elif prompt_type == "7sb_pattern":
        get_prompt_fn = get_7sb_pattern_prompt_fn()
    elif prompt_type == "pattern4":
        get_prompt_fn = get_pattern4_prompt_fn()
    else:
        raise ValueError()
    return get_prompt_fn


def get_prompt_factory(sb_list_w_pattern):
    instruction_d = {}
    for sb in sb_list_w_pattern:
        pattern_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_g_sel", f"{sb}.json")
        patterns = json.load(open(pattern_path, "r"))
        instruction_d[sb] = get_pattern_instruction(sb, patterns)

    all_sb_list = load_subreddit_list()
    for sb in all_sb_list:
        if sb not in sb_list_w_pattern:
            instruction_d[sb] = get_no_rule_instruction(sb)

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        print(inst)
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt


def get_7sb_pattern_prompt_fn():
    sb_list = [
        "Android", "fantasyfootball", "space", "TwoXChromosomes",
        "askscience", "pokemontrades", "TheSilphRoad"
    ]
    get_prompt = get_prompt_factory(sb_list)
    return get_prompt

def get_7sb_pattern3_prompt_fn():
    train_sb_list = [
        "Android", "fantasyfootball", "space", "TwoXChromosomes",
        "askscience", "pokemontrades", "TheSilphRoad"
    ]
    val_sb_list = [
        "SuicideWatch"
    ]
    sb_list = train_sb_list + val_sb_list
    get_prompt = get_prompt_factory(sb_list)
    return get_prompt


def get_pattern4_prompt_fn():
    instruction_d = {}
    all_sb_list = load_subreddit_list()
    found_list = []
    for sb in all_sb_list:
        pattern_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_g_sel", f"{sb}.json")
        if os.path.exists(pattern_path):
            try:
                patterns = json.load(open(pattern_path, "r"))
            except JSONDecodeError:
                print(pattern_path)
                raise
            instruction_d[sb] = get_pattern_instruction2(sb, patterns)
            found_list.append(sb)
        else:
            inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
            inst += f"Answer Yes or No, as a single token.\n"
            instruction_d[sb] = inst

    print("Sb with patterns: ", found_list)
    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"<Instruction>{inst}</Instruction>\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt



def get_pattern_prompt_fn_w_prepost(prefix, postfix):
    train_sb_list = [
        "Android", "fantasyfootball", "space", "TwoXChromosomes",
        "askscience", "pokemontrades", "TheSilphRoad"
    ]
    val_sb_list = [
        "SuicideWatch"
    ]
    sb_list_w_pattern = train_sb_list + val_sb_list
    instruction_d = {}
    for sb in sb_list_w_pattern:
        pattern_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_g_sel", f"{sb}.json")
        patterns = json.load(open(pattern_path, "r"))
        instruction_d[sb] = get_pattern_instruction_w_prepost(sb, prefix, postfix, patterns)

    all_sb_list = load_subreddit_list()
    for sb in all_sb_list:
        if sb not in sb_list_w_pattern:
            inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
            inst += f"Answer Yes or No, as a single token.\n"
            instruction_d[sb] = inst

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"<Instruction>{inst}</Instruction>\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt
