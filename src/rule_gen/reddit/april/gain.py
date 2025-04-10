import random
from collections import defaultdict

import fire

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import load_jsonl
from desk_util.path_helper import get_feature_pred_save_path, load_clf_pred
from rule_gen.reddit.april.inspect_apply import parse_per_feature_outputs
from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_q_lists
# Goal
# check if "features"/llama2_rp_cpq2 can improve over bert2_mix_sel
# implement algorithms to quantify each rules value in this dataset
from desk_util.runnable.run_eval import load_labels, clf_eval


import fire


def main(sb="fantasyfootball"):
    dataset = f"{sb}_2_val_100"
    base_run_name = "bert2_train_mix_sel"
    preds: list[tuple[str, int, float]] = load_clf_pred(dataset, base_run_name)
    labels = load_labels(dataset)
    base_score_d = clf_eval(preds, labels)
    print_metrics = list(base_score_d.keys())
    print("Base score:")
    table = [["q"] + print_metrics]
    row = ["init"]
    def pretty(item):
        if isinstance(item, float):
            return "{0:.2f}".format(item)
        else:
            return item
    for metric in print_metrics:
        print(f"{metric}\t{base_score_d[metric]}")
        row.append(pretty(base_score_d[metric]))
    table.append(row)

    run_name = f"llama2_rp_cpq2_{sb}"
    payload = load_csv_dataset_by_name(dataset)
    save_path = get_feature_pred_save_path(run_name, dataset)
    j_list = load_jsonl(save_path)
    tokens = run_name.split("_")
    engiFne_name = tokens[0]
    text_name = tokens[1]
    short_name = tokens[2]

    q_list = load_q_lists(text_name, short_name, sb)
    print(q_list)
    per_q = parse_per_feature_outputs(j_list, q_list)
    def rand_label():
        return bool(random.randint(0, 1))

    per_q["dummy"] = [(e[0], rand_label(), e[2]) for e in preds]

    t = 0.9
    for q_idx, entries in per_q.items():
        new_preds = []
        for data_idx in range(len(preds)):
            data_id, base_pred, score = preds[data_idx]
            data_id2, statement_pred, score2 = entries[data_idx]
            statement_pred = score2 > t

            new_pred = base_pred or statement_pred
            new_score = score + score2
            new_preds.append((data_id, new_pred, new_score))


        new_score_d = clf_eval(new_preds, labels)
        change_d = {}
        for key in new_score_d:
            s_before = base_score_d[key]
            if isinstance(s_before, float):
                s_after = new_score_d[key]
                change_d[key] = s_after - s_before

        # print("q_idx = {}".format(q_idx))
        row = [q_idx]
        for metric in print_metrics:
            row.append(pretty(new_score_d[metric]))
        table.append(row)

    print_table(table)


if __name__ == "__main__":
    fire.Fire(main)