import time
from collections import Counter

import tqdm
import json

from tqdm import tqdm
import os
from typing import Callable, Iterable

from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import read_csv, save_jsonl, read_jsonl
from desk_util.path_helper import get_feature_pred_save_path
from llama_user.llama_helper.get_llm_engine import get_llm_engine_predict_fn
from rule_gen.cpath import output_root_path



def assert_type_list_of_str(l: list[str]):
    for t in l:
        if not isinstance(t, str):
            raise AssertionError("Not a list of string, got list of {}".format(type(t)))


def load_rule_processed_json(run_name, sb):
    save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                             f"{run_name}", f"bert2_{sb}.json")
    return json.load(open(save_path, "r"))


class SpeedCounter:
    def __init__(self):
        self.counter = Counter()
        self.st = None

    def start(self):
        self.st = time.time()

    def end(self):
        ed = time.time()
        st = self.st
        print("Elapased={0:.2f}".format(ed - st))
        if ed - st < 0.15:
            self.counter["under 150ms"] += 1
        elif ed - ed > 1:
            self.counter["over 1000ms"] += 1
            print(self.counter)
        else:
            self.counter["150ms < t <= 1000ms"] += 1
            print(self.counter)


def get_apply_fn(run_name) -> Callable[[str], dict | list | str]:
    tokens = run_name.split("_")
    engine_name = tokens[0]
    api_fn = get_llm_engine_predict_fn(engine_name)
    text_name = tokens[1]
    if text_name == "rp": #  rp = rule_processing
        short_name = tokens[2]
        sb = tokens[3]
        d = {
            "cq": "cluster_questions",
            "cpq": "cluster_probe_questions",
        }
        rule_name = d[short_name]
        q_list: list[str] = load_rule_processed_json(rule_name, sb)
        assert_type_list_of_str(q_list)
    else:
        raise KeyError(text_name)

    def get_prompt(question, text) -> str:
        prompt = f"<text> {text} </text>"
        prompt += f"\n <instruction> {question} Output as Yes/No.</instruction>"
        return prompt

    max_text_len = 5000
    def predict(text):
        assert isinstance(text, str)
        text = text[:max_text_len]
        output = []
        for q in q_list:
            ret = api_fn(get_prompt(q, text))
            output.append(ret)
        return output

    return predict


def apply_fn_to_dataset(
    dataset, run_name,
    predict: Callable[[str], dict|list|str],
    overwrite_existing=False,
):
    payload = load_csv_dataset_by_name(dataset)
    save_path = get_feature_pred_save_path(run_name, dataset)
    if not overwrite_existing and os.path.exists(save_path):
        if len(read_csv(save_path)) == len(payload):
            print(f"Prediction exists. Skip prediction")
            print(f": {save_path}")
            return
        else:
            print(f"Prediction exists but not complete. Overwritting")
            print(f": {save_path}")

    def predict_wrap(item):
        data_id, text = item
        result = predict(text)
        return {"data_id": data_id, "result": result}

    pred_itr: Iterable = map(predict_wrap, tqdm(payload, desc="Processing", unit="item"))
    save_jsonl(pred_itr, save_path)
    print(f"Saved at {save_path}")


def load_feature_pred(run_name, dataset):
    save_path = get_feature_pred_save_path(run_name, dataset)
    predictions: list[dict] = read_jsonl(save_path)
    return predictions
