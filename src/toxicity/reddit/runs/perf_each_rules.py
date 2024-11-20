import math
import os

from chair.misc_lib import make_parent_exists
from chair.tab_print import print_table
from toxicity.cpath import output_root_path
from toxicity.io_helper import save_csv
from toxicity.path_helper import load_clf_pred, get_n_rules
from toxicity.runnable.run_eval import load_labels, clf_eval


def each_rule_results(sb):
    role = "detail"
    dataset = f"{sb}_val_100"
    file_name = f"sr2_{dataset}"
    labels = load_labels(dataset)

    columns = ["adj_f1", "precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    n_rule = get_n_rules(sb)
    for rule_idx in range(n_rule):
        run_name = f"api_sr_{sb}_{rule_idx}_{role}"
        run_name = f"api_sq_{sb}_{rule_idx}"

        print(run_name)
        preds = load_clf_pred(dataset, run_name)
        score_d = clf_eval(preds, labels)

        row = get_metrics_as_row(columns, score_d)

        table.append([str(rule_idx)] + row)

    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{file_name}.csv")
    make_parent_exists(save_path)
    print_table(table)
    save_csv(table, save_path)


def each_sq_results(sb):
    dataset = f"{sb}_val_100"
    file_name = f"sq_{dataset}"
    labels = load_labels(dataset)

    columns = ["adj_f1", "precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    n_rule = 21
    for rule_idx in range(n_rule):
        run_name = f"api_sq_{rule_idx}"

        print(run_name)
        preds = load_clf_pred(dataset, run_name)
        score_d = clf_eval(preds, labels)

        row = get_metrics_as_row(columns, score_d)

        table.append([str(rule_idx)] + row)

    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{file_name}.csv")
    make_parent_exists(save_path)
    print_table(table)
    save_csv(table, save_path)


def get_metrics_as_row(columns, score_d):
    score_d["tp"] = score_d["confusion_matrix"][1][1]
    score_d["fn"] = score_d["confusion_matrix"][1][0]
    score_d["fp"] = score_d["confusion_matrix"][0][1]
    score_d["tn"] = score_d["confusion_matrix"][0][0]

    P = score_d["precision"]
    R = score_d["recall"]
    k = 10
    adj_f1 = 2 * P * R / (P + k * R - k * P * R + R)
    score_d["adj_f1"] = adj_f1
    assert math.isclose(score_d["precision"], score_d["tp"] / (score_d["tp"] + score_d["fp"]))
    assert math.isclose(score_d["recall"], score_d["tp"] / (score_d["tp"] + score_d["fn"]))
    row = []
    for c in columns:
        row.append(score_d.get(c))
    return row


def main():
    rule_sb = "churning"
    each_sq_results(rule_sb)


if __name__ == "__main__":
    main()
