import math
import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list
from desk_util.runnable.run_eval import load_labels, clf_eval


def single_rule_result(rule_idx, rule_sb):
    run_name = f"api_sr_{rule_sb}_{rule_idx}_detail"
    single_run_result(run_name)


def single_run_result(run_name):
    subreddit_list = get_split_subreddit_list("train")
    print(run_name)
    columns = ["precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    for sb in subreddit_list:
        dataset = f"{sb}_val_100"
        preds = load_clf_pred(dataset, run_name)
        labels = load_labels(dataset)
        score_d = clf_eval(preds, labels)

        score_d["tp"] = score_d["confusion_matrix"][1][1]
        score_d["fn"] = score_d["confusion_matrix"][1][0]
        score_d["fp"] = score_d["confusion_matrix"][0][1]
        score_d["tn"] = score_d["confusion_matrix"][1][1]
        math.isclose(score_d["precision"], score_d["tp"] / (score_d["tp"] + score_d["fp"]))
        math.isclose(score_d["recall"], score_d["tp"] / (score_d["tp"] + score_d["fn"]))

        row = [sb]
        for c in columns:
            row.append(score_d.get(c))

        table.append(row)
    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{run_name}.csv")
    make_parent_exists(save_path)
    save_csv(table, save_path)


def main():
    rule_sb = "TwoXChromosomes"
    rule_idx = 0
    single_rule_result(rule_idx, rule_sb)


if __name__ == "__main__":
    main()
