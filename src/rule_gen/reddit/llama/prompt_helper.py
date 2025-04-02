import json
import os
from json import JSONDecodeError

from rule_gen.cpath import output_root_path
from rule_gen.reddit.classifier_loader.inst_builder import get_instruction_from_run_name, get_no_rule_instruction
from rule_gen.reddit.llama.train_prompt_gen import get_pattern_instruction, get_pattern_instruction2, \
    get_pattern_instruction_w_prepost
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_subreddit_list, get_rp_path


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


def get_prompt_factory_no_rule2(max_len):
    sb_list = load_subreddit_list()
    instruction_d = {}
    for sb in sb_list:
        run_name = f"api_{sb}_both"
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        instruction = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
        instruction_d[sb] = instruction

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        if max_len is not None:
            text = text[:max_len]
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>\n"
        return prompt
    return get_prompt


def get_prompt_factory_strict():
    sb_list = load_subreddit_list()
    instruction_d = {}
    for sb in sb_list:
        run_name = f"api_{sb}_both"
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        role = tokens[-1]
        inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
        inst += f"Note that {sb} has very strict moderation criteria.\n"
        inst += f"Answer Yes or No, as a single token.\n"
        instruction_d[sb] = inst

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt


def get_prompt_factory_generous():
    sb_list = load_subreddit_list()
    instruction_d = {}
    for sb in sb_list:
        run_name = f"api_{sb}_both"
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
        inst += f"Note that the {sb} subreddit has very generous moderation criteria.\n"
        inst += f"Answer Yes or No, as a single token.\n"
        instruction_d[sb] = inst

    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"{inst}\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt


def get_prompt_factory_say_why():
    sb_list = load_subreddit_list()
    instruction_d = {}
    for sb in sb_list:
        run_name = f"api_{sb}_both"
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
        inst += f"Explain why it is predicted so (even heuristic).\n"
        inst += f"Finish the response with '#### Yes' or '#### No'\n"
        instruction_d[sb] = inst

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
    elif prompt_type == "sb_name2":
        get_prompt_fn = get_prompt_factory_no_rule2()
    elif prompt_type == "sb_name_strict":
        get_prompt_fn = get_prompt_factory_strict()
    elif prompt_type == "sb_name_generous":
        get_prompt_fn = get_prompt_factory_generous()
    elif prompt_type == "sb_name_say_why":
        get_prompt_fn = get_prompt_factory_say_why()
    elif prompt_type == "7sb_pattern":
        get_prompt_fn = get_7sb_pattern_prompt_fn()
    elif prompt_type == "pattern4":
        get_prompt_fn = get_pattern4_prompt_fn()
    elif prompt_type == "pattern_f":
        get_prompt_fn = get_pattern_f_prompt_fn()
    elif prompt_type == "pattern_g":
        get_prompt_fn = get_pattern_g_prompt_fn()
    elif prompt_type == "pattern_h":
        get_prompt_fn = get_pattern_h_prompt_fn()
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


def get_pattern_f_prompt_fn():
    instruction_d = {}
    all_sb_list = load_subreddit_list()

    inst_first_line_fmt =  "If the following text is posted in {} subreddit, removed by a moderator or AutoMod based on subreddit rules?\n"
    inst_line_line = f"\nOutput only 'Yes' or 'No' as a single token, without explanation.\n"
    def format_prompt(sb, patterns):
        inst = inst_first_line_fmt.format(sb)
        inst += "Here are common patterns that are deleted, with the key part makred with <reason>: \n" \
                "<Patterns>\n"
        pattern_str = "\n".join(patterns)
        inst += pattern_str + "\n</Patterns>"
        inst += "\nNote that there could be moderated texts which are not captured by these patterns.\n"
        inst += inst_line_line
        return inst

    found_list = []
    not_found_list = []
    for sb in all_sb_list:
        pattern_path = get_rp_path("ngram_93_rule1", f"{sb}.json")
        try:
            patterns = json.load(open(pattern_path, "r"))
        except FileNotFoundError:
            patterns = []
        except JSONDecodeError:
            print(pattern_path)
            raise

        if patterns:
            instruction_d[sb] = format_prompt(sb, patterns)
            found_list.append(sb)
        else:
            inst = inst_first_line_fmt.format(sb)
            inst += inst_line_line
            instruction_d[sb] = inst
            not_found_list.append(sb)

    print("Sb with patterns: ", found_list)
    print("Sb without patterns: ", not_found_list)
    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"<Instruction>{inst}</Instruction>\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt

def get_pattern_g_prompt_fn():
    instruction_d = {}
    all_sb_list = load_subreddit_list()
    inst_first_line_fmt =  "If the following text is posted in {} subreddit, removed by a moderator or AutoMod based on subreddit rules?\n"

    def format_prompt(sb, patterns):
        inst = inst_first_line_fmt.format(sb)
        inst += "Here are common patterns that are deleted, with the key part marked with <reason>: \n" \
                "<Patterns>\n"

        pattern_str_list = [f"<text> {t} </text>" for t in patterns]
        pattern_str = "\n".join(pattern_str_list)
        inst += pattern_str + "\n</Patterns>"
        inst += "\nNote that there could be moderated texts which are not captured by these patterns."
        inst += f"\n    Answer Yes or No, as a single token.\n"
        return inst

    found_list = []
    not_found_list = []
    for sb in all_sb_list:
        pattern_path = get_rp_path("ngram_93_rule1", f"{sb}.json")
        try:
            patterns = json.load(open(pattern_path, "r"))
        except FileNotFoundError:
            patterns = []
        except JSONDecodeError:
            print(pattern_path)
            raise

        if patterns:
            instruction_d[sb] = format_prompt(sb, patterns)
            found_list.append(sb)
        else:
            inst = inst_first_line_fmt.format(sb)
            inst += f"Answer Yes or No, as a single token.\n"
            instruction_d[sb] = inst
            not_found_list.append(sb)

    print("Sb with patterns: ", found_list)
    print("Sb without patterns: ", not_found_list)
    def get_prompt(text, sb):
        inst = instruction_d[sb]
        prompt = f"<Instruction>{inst}</Instruction>\n<BEGIN TEXT>{text}\n<END TEXT>"
        return prompt
    return get_prompt


def get_pattern_h_prompt_fn():
    instruction_d = {}
    all_sb_list = load_subreddit_list()
    inst_first_line_fmt =  "If the following text is posted in {} subreddit, removed by a moderator or AutoMod based on subreddit rules?\n"

    def format_prompt(sb, patterns):
        inst = inst_first_line_fmt.format(sb)
        inst += "Here are common patterns that are deleted, with the key part marked with <reason>: \n" \
                "<Patterns>\n"

        def modify(pattern):
            pattern = pattern.replace("<reason>", "<reason> ")
            pattern = pattern.replace("</reason>", " </reason>")
            return pattern

        patterns = map(modify, patterns)
        pattern_str_list = [f"<pattern> {t} </pattern>" for t in patterns]
        pattern_str = "\n".join(pattern_str_list)
        inst += pattern_str + "\n</Patterns>"
        inst += "\nNote that there could be moderated texts which are not captured by these patterns."
        inst += f"\n    Answer Yes or No, as a single token.\n"
        return inst

    found_list = []
    not_found_list = []
    for sb in all_sb_list:
        pattern_path = get_rp_path("ngram_93_rule1", f"{sb}.json")
        try:
            patterns = json.load(open(pattern_path, "r"))
        except FileNotFoundError:
            patterns = []
        except JSONDecodeError:
            print(pattern_path)
            raise

        if patterns:
            instruction_d[sb] = format_prompt(sb, patterns)
            found_list.append(sb)
        else:
            inst = inst_first_line_fmt.format(sb)
            inst += f"Answer Yes or No, as a single token.\n"
            instruction_d[sb] = inst
            not_found_list.append(sb)

    print("Sb with patterns: ", found_list)
    print("Sb without patterns: ", not_found_list)
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
