import json
import os

from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_reddit_rule_path, get_reddit_manual_prompt_path, \
    get_reddit_auto_prompt_path, get_reddit_rule_path2


def get_paraphrased_instruction(run_name, pos_keyword):
    # api_srr_sb_1
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:-1])
    rule_idx = int(tokens[-1])
    rule_save_path = get_reddit_para_rule_path(sb)
    rules = [t.strip() for t in open(rule_save_path, "r").readlines()]
    rule_text = rules[rule_idx]
    inst_summary = f"If so, output '{pos_keyword}' as a first token. If not, output 'no'"
    return rule_text + "\n " + inst_summary


def get_rule_showing_instruction(sb, role):
    pos_keyword = "unsafe"
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    if role == "both":
        rule_text = " ".join([r["summary"] + ". " + r["detail"] for r in rules])
    else:
        rule_text = " ".join([r[role] for r in rules])
    inst_summary = "The above rule describes prohibited contents. Classify if the following text is prohibited. "
    inst_summary += f"If prohibited, output '{pos_keyword}' as a first token. If not, output 'safe'"
    return rule_text + "\n " + inst_summary, pos_keyword


def get_rule2_from_name(run_name):
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:])
    return get_rule_showing_instruction2(sb)


def get_v2_style_instruction(run_name):
    prefix = "api_v2_"
    name = run_name[len(prefix):]
    last_token = name.split("_")[-1]
    idx = int(last_token)
    file_name = name[:-len(last_token)-1]
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rule_processing", f"{file_name}.json")
    rule_text = json.load(open(rule_save_path, "r"))[idx]
    pos_keyword = "yes"
    max_len = 5000
    if file_name == "generic_norm":
        def template_fn(text):
            text = text[:max_len]
            prompt = "Does the following text violate the rule? "
            prompt += f"If it violates, output '{pos_keyword}' as a first token. If not, output 'no'"
            prompt += f"<rule> {rule_text} </rule>"
            prompt += f"<text> {text} </text>"
            return prompt
    else:
        def template_fn(text):
            text = text[:max_len]
            prompt = f"{rule_text} Output as Yes/No."
            prompt += f"\n <text> {text} </text>"
            return prompt

    return template_fn, pos_keyword


def get_rule_showing_instruction2(sb):
    pos_keyword = "unsafe"
    rule_save_path = get_reddit_rule_path2(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text_list = []
    for r in rules:
        if r['kind'] in ["comment", "all"]:
            rule_text = r["short_name"] + ". " + r["description"]
            rule_text_list.append(rule_text)
    rule_text = " ".join(rule_text_list)
    inst_summary = "The above rule describes prohibited contents. Classify if the following text is prohibited. "
    inst_summary += f"If prohibited, output '{pos_keyword}' as a first token. If not, output 'safe'"
    return rule_text + "\n " + inst_summary, pos_keyword


def get_no_rule_instruction(run_name):
    tokens = run_name.split("_")
    sb = "_".join(tokens[1:-1])
    role = tokens[-1]
    inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
    inst += f"Answer Yes or No, as a single token.\n"
    return inst, "Yes"


def get_sb_agnostic():
    pos_keyword = "Yes"
    inst = f"If the following text is posted on Reddit, will it be moderated (deleted)?"
    inst += f"Answer Yes or No, as a single token. \n"
    return inst, pos_keyword


def get_single_rule_instruction_by_name(run_name, pos_keyword):
    # api_sr_sb_1_detail
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:-2])
    rule_idx = int(tokens[-2])
    role = tokens[-1]
    assert role in ["summary", "detail", "both"]
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    if role == "both":
        r = rules[rule_idx]
        rule_text = r["summary"] + ". " + r["detail"]
    else:
        rule_text = rules[rule_idx][role]

    inst_summary = "Does the following text violate the rule above? "
    inst_summary += f"If so, output '{pos_keyword}' as a first token. If not, output 'no'"
    return rule_text + "\n " + inst_summary


def get_single_rule_instruction2(run_name, pos_keyword):
    # api_sr2_sb_1_detail
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:-2])
    rule_idx = int(tokens[-2])
    role = tokens[-1]
    assert role in ["summary", "detail", "both"]
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    if role == "both":
        r = rules[rule_idx]
        rule_text = r["summary"] + ". " + r["detail"]
    else:
        rule_text = rules[rule_idx][role]

    inst_summary = "Does the following text violate the rule above? "
    inst_summary += f"If so, output '{pos_keyword}' as a first token. If not, output 'no'. If unclear, output 'no'."
    return rule_text + "\n " + inst_summary


def get_manual_instruction(name, pos_keyword):
    path = get_reddit_manual_prompt_path(name)
    prompt = open(path, "r").read()
    prompt += f"If so, output '{pos_keyword}' as a first token. If not, output 'no'"
    return prompt


def get_autogen_instruction(name, pos_keyword):
    typename, sbname = name.split("_")
    path = get_reddit_auto_prompt_path(typename, sbname)
    prompt = open(path, "r").read()
    prompt += "\nConsidering the rules above will the following text be deleted?"
    prompt += f"If so, output '{pos_keyword}' as a first token. If not, output 'no'"
    return prompt


def get_instruction_from_run_name(run_name):
    if run_name.startswith("api_sr_"):
        pos_keyword = "yes"
        instruction = get_single_rule_instruction_by_name(run_name, pos_keyword)
    elif run_name.startswith("api_sr2_"):
        pos_keyword = "yes"
        instruction = get_single_rule_instruction2(run_name, pos_keyword)
    elif run_name.startswith("api_srr_"):
        pos_keyword = "yes"
        instruction = get_paraphrased_instruction(run_name, pos_keyword)
    elif run_name.startswith("api_man_"):
        postfix = run_name[len("api_man_"):]
        pos_keyword = "yes"
        instruction = get_manual_instruction(postfix, pos_keyword)
    elif run_name.startswith("api_cq"):
        postfix = run_name[len("api_cq"):]
        pos_keyword = "yes"
        instruction = get_manual_instruction(postfix, pos_keyword)
    elif run_name.startswith("api_auto_"):
        postfix = run_name[len("api_auto_"):]
        pos_keyword = "yes"
        instruction = get_autogen_instruction(postfix, pos_keyword)
    elif run_name.startswith("api_rule2_"):
        instruction, pos_keyword = get_rule2_from_name(run_name)
    elif run_name.startswith("api_none"):
        instruction, pos_keyword = get_sb_agnostic()
    elif run_name.startswith("api_v2"):
        instruction, pos_keyword = get_v2_style_instruction(run_name)
    else:
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        role = tokens[-1]
        if role in ["summary", "detail", "both"]:
            try:
                instruction, pos_keyword = get_rule_showing_instruction(sb, role)
            except FileNotFoundError:
                print("Rules does not exist for {}. Run with no rule".format(sb))
                instruction, pos_keyword = get_no_rule_instruction(run_name)
        elif role == "none":
            instruction, pos_keyword = get_no_rule_instruction(run_name)
        else:
            raise ValueError()

    return instruction, pos_keyword


def get_reddit_para_rule_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules_re", f"{sb}.txt")
    return rule_save_path
