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
from rule_gen.reddit.s9.clf_for_one import compute_f_k
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


def convert_score(score, term):
    # prob = np.exp(score)
    if term == "Yes" or term == "yes":
        return score
    else:
        return -3
        # prob = 0
    # return prob


def convert_score_discrete(score, term):
    if term == "Yes" or term == "yes":
        return 1
    else:
        return 0


def main():
    table = []
    s9_run_name = "llama_s9_{}"
    s9_run_name = "llama_s9nosb"
    feature_name2 = ""
    print("load_run_name_fmt", s9_run_name)
    # sb_list = get_split_subreddit_list("train")
    f_name = "table"
    sb_list = get_split_subreddit_list("val")
    def load_here(dataset):
        X = load_s9(dataset, s9_run_name, target_seq, convert_score_fn)
        # x_i = load_feature_csv(train_dataset, f_name)
        print((X))
        labels = load_labels(dataset)
        y = right(labels)
        return X, y

    target_seq = [f"S{i}" for i in range(1, 10)]
    proxy = get_task_manager_proxy()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    table2 = []
    for sb in sb_list:
        try:
            load_run_name = s9_run_name.format(sb)
            train_dataset = f"{sb}_2_train_100"
            val_dataset = f"{sb}_2_val_100"
            report_run_name = load_run_name + "_100d"
            convert_score_fn = convert_score
            X_train, y_train = load_here(f"{sb}_2_train_100")
            X_test, y_test = load_here(val_dataset)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_score = f1_score(y_train, train_pred)
            val_pred = clf.predict(X_test)
            val_score = f1_score(y_test, val_pred)
            f19 = compute_f_k(y_test, val_pred, 19)
            row = [sb, val_score, f19]
            table.append(row)
            row2 = [sb, val_score]
            for t in clf.coef_[0]:
                row2.append("{:.3f}".format(t))

            row2.append("{:.3f}".format(clf.intercept_[0]))
            table2.append(row2)

            # print(f"\nResults for subreddit: {sb}")
            # print(f"Training F1 Score: {train_score:.4f}")
            # print(f"Validation F1 Score: {val_score:.4f}")
            # print("coef", clf.coef_, clf.intercept_)
            # proxy.report_number(report_run_name, val_score, val_dataset, "f1")
        except FileNotFoundError as e:
            print(e)

    print_table(table)
    print_table(table2)


if __name__ == "__main__":
    main()




