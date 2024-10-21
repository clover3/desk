import json
import os
import random

from toxicity.cpath import output_root_path
from toxicity.llama_helper.lf_client import LLMClient
from toxicity.reddit.path_helper import get_reddit_rule_path


def load_api_based(run_name):
    client = LLMClient(max_prompt_len=5000)
    if run_name.startswith("api_sr_"):
        pos_keyword = "yes"
        instruction = get_single_rule_instruction_by_name(run_name, pos_keyword)
    if run_name.startswith("api_srr_"):
        pos_keyword = "yes"
        instruction = get_paraphrased_instruction(run_name, pos_keyword)
    else:
        pos_keyword = "unsafe"
        instruction = get_instruction_by_name(run_name, pos_keyword)

    def predict(text):
        ret_text = client.ask(text, instruction)
        pred = pos_keyword in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def get_random_classifier():
    def predict(text):
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


def get_instruction_by_name(run_name, pos_keyword):
    tokens = run_name.split("_")
    sb = "_".join(tokens[1:-1])
    role = tokens[-1]
    assert role in ["summary", "detail"]
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text = " ".join([r[role] for r in rules])
    inst_summary = "The above rule describes prohibited contents. Classify if the following text is prohibited. "
    inst_summary += f"If prohibited, output '{pos_keyword}' as a first token. If not, output 'safe'"
    return rule_text + "\n " + inst_summary


def get_single_rule_instruction_by_name(run_name, pos_keyword):
    # api_sr_sb_1_detail
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:-2])
    rule_idx = int(tokens[-2])
    role = tokens[-1]
    assert role in ["summary", "detail"]
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text = rules[rule_idx][role]
    inst_summary = "Does the following text violate the rule above? "
    inst_summary += f"If so, output '{pos_keyword}' as a first token. If not, output 'no'"
    return rule_text + "\n " + inst_summary


def get_reddit_para_rule_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules_re", f"{sb}.txt")
    return rule_save_path


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


def get_classifier(run_name):
    if run_name.startswith("bert"):
        from toxicity.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
        return get_classifier_pipeline(run_name)
    elif run_name == "random":
        return get_random_classifier()
    elif run_name.startswith("api_"):
        return load_api_based(run_name)
    else:
        raise ValueError(f"{run_name} is not expected")
