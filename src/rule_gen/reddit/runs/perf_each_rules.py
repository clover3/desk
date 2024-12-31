import os

import fire

from chair.misc_lib import make_parent_exists
from chair.tab_print import print_table
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_n_rules
from desk_util.io_helper import save_csv
from desk_util.path_helper import load_clf_pred
from desk_util.runnable.run_eval import load_labels, clf_eval



def each_sq_results(sb):
    dataset = f"{sb}_val_100"
    run_name_head = f"chatgpt_sr_"
    n_rule = 8
    file_name = run_name_head + f"_{dataset}"
    labels = load_labels(dataset)

    columns = ["adj_f1", "precision", "recall", "tp", "fn", "fp", "tn"]
    head = [""] + columns
    table = [head]
    for rule_idx in range(n_rule):
        run_name = run_name_head + str(rule_idx)
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
    n_all = score_d["tp"] + score_d["fn"] + score_d["fp"] + score_d["tn"]
    score_d["true_rate"] = (score_d["tp"] + score_d["fp"]) / n_all
    P = score_d["precision"]
    R = score_d["recall"]
    k = 10
    adj_f1 = 2 * P * R / (P + k * R - k * P * R + R)
    score_d["adj_f1"] = adj_f1
    row = []
    for c in columns:
        row.append(score_d.get(c))
    return row


def each_rule_results(sb="churning"):
    run_name_head = "chatgpt_sr"
    role = "both"
    dataset = f"{sb}_val_100"
    save_name = f"{run_name_head}_{dataset}"
    labels = load_labels(dataset)
    columns = ["adj_f1", "precision", "recall", "tp", "fn", "fp", "tn", "true_rate"]
    head = [""] + columns
    table = [head]
    n_rule = get_n_rules(sb)
    for rule_idx in range(n_rule):
        run_name = f"{run_name_head}_{sb}_{rule_idx}_{role}"
        print(run_name)
        preds = load_clf_pred(dataset, run_name)
        score_d = clf_eval(preds, labels)
        row = get_metrics_as_row(columns, score_d)
        table.append([str(rule_idx)] + row)

    save_path = os.path.join(output_root_path, "reddit", "analysis", f"{save_name}.csv")
    make_parent_exists(save_path)
    print_table(table)
    save_csv(table, save_path)


if __name__ == "__main__":
    fire.Fire(each_rule_results)