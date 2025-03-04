from collections import Counter

import fire
import json

from desk_util.io_helper import read_csv
from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.single_run2.info_gain import rule_value


def get_negs(sb):
    data_name = "train_data2"
    train_dataset = read_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))

    for text, label in train_dataset:
        if int(label) == 0:
            yield text


def main(sb= "TwoXChromosomes"):
    client = LLMClient(max_prompt_len=5000)
    # core_instruction = input("Enter prompt: ")
    core_instruction = "Is this text asking for advice on trades, who to start or bench, or waiver wire decisions?"
    instruction = core_instruction
    instruction += "\n Answer yes/no.\n"
    min_sel_diff_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel", "", f"{sb}.csv")
    todo = json.load(open(min_sel_diff_path, "r"))
    counter = Counter()
    columns = ["TP", "FP", "FN", "TN"]
    print("\t".join(columns))
    for item in todo:
        label = item["domain_pred"]
        text_j = {"text": item["text"]}
        if int(item["tms_pred"]):
            pred = True
        else:
            prompt = f"{instruction}\n{json.dumps(text_j)}"
            res = client.ask(prompt)
            pred = "yes" in res.lower()
        case = {
            (True, True): "TP",
            (True, False): "FP",
            (False, True): "FN",
            (False, False): "TN",
        }[(pred, label)]
        counter[case] += 1
        s = ", ".join([str(counter[case]) for case in columns])
        print("\r" + s, end='\r')
    print()


def main(sb= "TwoXChromosomes"):
    client = LLMClient(max_prompt_len=5000)
    # core_instruction = input("Enter prompt: ")
    # core_instruction = "Is this text asking for advice on trades, who to start or bench, or waiver wire decisions?"
    core_instruction = "Does this text involve discussions about fantasy football"
    print(core_instruction)
    instruction = core_instruction
    instruction += "\n Answer yes/no.\n"
    min_sel_diff_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel", "", f"{sb}.csv")
    todo = json.load(open(min_sel_diff_path, "r"))
    counter = Counter()
    columns = ["TP", "FP", "FN", "TN"]
    print("\t".join(columns))
    preds = []
    labels = []
    for item in todo:
        label = item["domain_pred"]
        text_j = {"text": item["text"]}
        if int(item["tms_pred"]):
            continue

        prompt = f"{instruction}\n{json.dumps(text_j)}"
        res = client.ask(prompt)
        pred = "yes" in res.lower()
        case = {
            (True, True): "TP",
            (True, False): "FP",
            (False, True): "FN",
            (False, False): "TN",
        }[(pred, label)]
        preds.append(int(pred))
        labels.append(int(label))
        counter[case] += 1
        s = ", ".join([str(counter[case]) for case in columns])
        print("\r" + s, end='\r')
    print()
    gain = rule_value(labels, preds)
    print("Gain", gain)


if __name__ == "__main__":
    fire.Fire(main)