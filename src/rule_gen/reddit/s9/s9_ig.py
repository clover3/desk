import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.criteria_checker.save_features import load_feature_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.s9.classifier import convert_score
from rule_gen.reddit.s9.clf_for_one import compute_f_k, convert_score_discrete
from rule_gen.reddit.s9.feature_extractor import get_value
from taskman_client.task_proxy import get_task_manager_proxy


def load_s9(dataset, run_name, seq, convert_score_fn) -> np.array:
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "j_res", dataset, f"{run_name}.json")
    j = json.load(open(res_save_path))
    X = []
    for data_id, _, ret_text, output in j:
        d = {}
        for code, term, score in output:
            prob = convert_score_fn(score, term)
            d[code] = prob
        try:
            x = [d[item] for item in seq]
        except KeyError:
            x = [0] * len(seq)
        X.append(x)

    return np.array(X)


def main():
    table = []
    s9_run_name = "llama_s9nosb"
    print("load_run_name_fmt", s9_run_name)
    sb_list = get_split_subreddit_list("train")

    def load_here(dataset):
        X = load_s9(dataset, s9_run_name, target_seq, convert_score_fn)
        labels = load_labels(dataset)
        y = right(labels)
        return X, y

    target_seq = [f"S{i}" for i in range(1, 10)]
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    for sb in sb_list:
        try:
            train_dataset = f"{sb}_2_train_100"
            val_dataset = f"{sb}_2_val_100"
            convert_score_fn = convert_score_discrete
            X_train, y_train = load_here(train_dataset)
            row = [sb]
            for i in range(1, 10):
                X_i = X_train[:, i-1]
                ret = get_value(y_train, X_i)
                row.append(ret)
            table.append(row)
        except FileNotFoundError as e:
            print(e)

    print_table(table)


if __name__ == "__main__":
    main()




