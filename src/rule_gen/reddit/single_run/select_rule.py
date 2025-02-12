import json
import os

import fire
import tqdm
from tqdm import tqdm

from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv, read_csv
from llama_user.llama_helper.lf_client import LLMClient
from desk_util.path_helper import get_dataset_pred_save_path
from rule_gen.reddit.path_helper import get_reddit_rule_path

def get_instruction_by_name(run_name):
    tokens = run_name.split("_")
    sb = "_".join(tokens[1:])
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))

    lines = []
    for rule_idx, r in enumerate(rules):
        line = r["summary"] + " " + r["detail"]
        if not line[0].isdigit():
            line = f"{rule_idx+1} {line}"
        lines.append(line)

    rule_text = " ".join(lines)
    inst_summary = "If the following text is deleted, which of the above rule is most appropriate? "
    inst_summary += f"Only output the number of the rule as the first token. "
    return rule_text + "\n " + inst_summary


def load_predictor(run_name):
    client = LLMClient(max_prompt_len=5000)
    if run_name.startswith("rs_"):
        instruction = get_instruction_by_name(run_name)
    else:
        assert False

    def predict(text):
        ret_text = client.ask(text, instruction)
        return ret_text, 0
    return predict


def run_seq2seq(dataset, run_name, predict_fn):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    def predict(e):
        id, text = e
        text, score = predict_fn(text)
        return id, text, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path: str = get_dataset_pred_save_path(run_name, dataset)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def main(dataset, run_name):
    run_seq2seq(dataset, run_name, load_predictor(run_name))


if __name__ == "__main__":
    fire.Fire(main)
