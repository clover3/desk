import json
import os
import random

from toxicity.cpath import output_root_path
from toxicity.llama_helper.lf_client import LLMClient
from toxicity.reddit.path_helper import get_reddit_rule_path
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator




def get_instruction_from_run_name(run_name):
    if run_name.startswith("api_sr_"):
        pos_keyword = "yes"
        instruction = get_single_rule_instruction_by_name(run_name, pos_keyword)
    elif run_name.startswith("api_srr_"):
        pos_keyword = "yes"
        instruction = get_paraphrased_instruction(run_name, pos_keyword)
    else:
        tokens = run_name.split("_")
        sb = "_".join(tokens[1:-1])
        role = tokens[-1]
        if role in ["summary", "detail", "both"]:
            instruction, pos_keyword = get_rule_showing_instruction(sb, role)
        elif role == "none":
            instruction, pos_keyword = get_no_rule_instruction(run_name)
        else:
            raise ValueError()

    return instruction, pos_keyword


def get_random_classifier():
    def predict(text):
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


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


def get_no_rule_instruction(run_name):
    tokens = run_name.split("_")
    sb = "_".join(tokens[1:-1])
    role = tokens[-1]
    inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?"
    inst += f"Answer Yes or No, as a single token.\n"
    return inst, "Yes"


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


class NumberAdder:
    def __init__(self, msg="Final number:"):
        self.number = 0
        self.msg = msg

    def add(self, value):
        self.number += value

    def __del__(self):
        print(f"{self.msg} {self.number}")


def dummy_counter(run_name):
    run_name = run_name[len("dummy_"):]
    instruction, pos_keyword = get_instruction_from_run_name(run_name)
    print(instruction)
    adder = NumberAdder()
    def predict(text):
        n_char = len(instruction) + len(text)
        n_char = min(n_char, 5000)
        adder.add(n_char)
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


def load_api_based(run_name):
    client = LLMClient(max_prompt_len=5000)
    instruction, pos_keyword = get_instruction_from_run_name(run_name)

    def predict(text):
        ret_text = client.ask(text, instruction)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def load_chatgpt_based(run_name) -> Callable[[str], tuple[int, float]]:
    from toxicity.apis.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")
    run_name = run_name.replace("chatgpt_", "api_")
    instruction, pos_keyword = get_instruction_from_run_name(run_name)
    max_prompt_len = 5000

    def predict(text):
        prompt = instruction + "\n" + text[:max_prompt_len]
        ret_text = client.request(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0

    return predict


def get_classifier(run_name) -> Callable[[str], tuple[int, float]]:
    if run_name.startswith("bert"):
        from toxicity.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
        return get_classifier_pipeline(run_name)
    elif run_name == "random":
        return get_random_classifier()
    elif run_name.startswith("dummy_"):
        return dummy_counter(run_name)
    elif run_name.startswith("api_"):
        return load_api_based(run_name)
    elif run_name.startswith("chatgpt_"):
        return load_chatgpt_based(run_name)

    else:
        raise ValueError(f"{run_name} is not expected")


gpt_prefix_list = ["chatgpt_", "gpt-4o_", "gpt-4o-mini"]


class PromptBuilder:
    def __init__(self, run_name):
        self.run_name = run_name
        self.max_text_len = 2000
        run_name_for_inst = None
        for prefix in gpt_prefix_list:
            if run_name.startswith(prefix):
                run_name_for_inst = run_name.replace(prefix, "api_")
                break

        if run_name_for_inst is None:
            raise ValueError()

        instruction, pos_keyword = get_instruction_from_run_name(run_name_for_inst)
        self.instruction = instruction
        self.pos_keyword = pos_keyword

    def get_prompt(self, text):
        prompt = self.instruction + "\n" + text[:self.max_text_len]
        return prompt

    def get_label_from_response(self, response):
        pred = self.pos_keyword in response.lower()
        ret = int(pred)
        return ret
