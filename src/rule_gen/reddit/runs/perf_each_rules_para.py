import math
import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_n_rules
from desk_util.io_helper import save_csv
from desk_util.path_helper import load_clf_pred
from desk_util.runnable.run_eval import load_labels, clf_eval


def each_rule_results(sb):
    dataset = f"{sb}_val_100"
    file_name = f"each_rules_r_{dataset}"
    labels = load_labels(dataset)

    columns = ["precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    n_rule = get_n_rules(sb)
    for rule_idx in range(n_rule):
        run_name = f"api_srr_{sb}_{rule_idx}"
        print(run_name)
        preds = load_clf_pred(dataset, run_name)
        score_d = clf_eval(preds, labels)

        row = get_metrics_as_row(columns, sb, score_d)

        table.append(row)

    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{file_name}.csv")
    make_parent_exists(save_path)
    save_csv(table, save_path)


def get_metrics_as_row(columns, sb, score_d):
    score_d["tp"] = score_d["confusion_matrix"][1][1]
    score_d["fn"] = score_d["confusion_matrix"][1][0]
    score_d["fp"] = score_d["confusion_matrix"][0][1]
    score_d["tn"] = score_d["confusion_matrix"][0][0]
    assert math.isclose(score_d["precision"], score_d["tp"] / (score_d["tp"] + score_d["fp"]))
    assert math.isclose(score_d["recall"], score_d["tp"] / (score_d["tp"] + score_d["fn"]))
    row = [sb]
    for c in columns:
        row.append(score_d.get(c))
    return row


def main(rule_sb="churning"):
    # rule_sb = "churning"
    # rule_sb = "TwoXChromosomes"
    each_rule_results(rule_sb)


if __name__ == "__main__":
    main()
