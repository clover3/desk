import math
import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list
from desk_util.runnable.run_eval import load_labels, clf_eval
import json
import fire

from sklearn.feature_selection import mutual_info_classif
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_rule_path, get_n_rules
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.runs.perf_each_rules import flatten_confusion_matrix


def rule_comb(sb):
    n_rule = get_n_rules(sb)
    rules = json.load(open(get_reddit_rule_path(sb), "r"))
    print(sb)
    rule_save_path = get_reddit_rule_path(sb)
    dataset = f"{sb}_val_100"
    save_name= "per_rule_" + dataset
    labels = load_labels(dataset)
    columns = ["rule", "f1", "precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    for rule_idx in range(n_rule):
        run_name = f"chatgpt_sr_{sb}_{rule_idx}_both"
        preds = load_clf_pred(dataset, run_name)
        score_d = clf_eval(preds, labels)
        flatten_confusion_matrix(score_d)
        score_d["rule"] = rules[rule_idx]["summary"]

        row = [f"{sb}_{rule_idx}"]
        for c in columns:
            row.append(score_d.get(c))

        table.append(row)
    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{save_name}.csv")
    make_parent_exists(save_path)
    save_csv(table, save_path)


def main(sb="TwoXChromosomes"):
    rule_comb(sb)


if __name__ == "__main__":
    fire.Fire(main)
