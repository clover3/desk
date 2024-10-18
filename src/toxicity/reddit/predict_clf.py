import json
import random

import torch
import fire

from toxicity.clf_util import clf_predict_w_predict_fn, clf_predict_w_batch_predict_fn
from toxicity.llama_helper.lf_client import LLMClient
from toxicity.path_helper import get_model_save_path
from toxicity.reddit.path_helper import get_reddit_rule_path
from toxicity.runnable.run_eval_clf import run_eval_clf
from transformers import pipeline

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_classifier_pipeline(run_name):
    model_path = get_model_save_path(run_name)
    pipe = pipeline("text-classification", model=model_path, device=get_device())

    label_map = {
        "LABEL_0": "0",
        "LABEL_1": "1",
    }

    def predict(text):
        r = pipe(text, truncation=True)[0]
        label = label_map[r["label"]]
        score = r["score"]
        if label == "0":
            score = -score
        return label, score

    def batch_predict(text_list):
        for r in pipe(text_list, truncation=True):
            label = label_map[r["label"]]
            score = r["score"]
            if label == "0":
                score = -score
            yield label, score

    return predict


def load_api_based(run_name):
    client = LLMClient(max_prompt_len=5000)
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


def get_classifier(run_name):
    if run_name.startswith("bert"):
        return get_classifier_pipeline(run_name)
    elif run_name == "random":
        return get_random_classifier()
    elif run_name.startswith("api_"):
        return load_api_based(run_name)
    else:
        raise ValueError(f"{run_name} is not expected")


def predict_clf_main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics=""
) -> None:
    predict_fn = get_classifier(run_name)
    clf_predict_w_predict_fn(dataset, run_name, predict_fn)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
