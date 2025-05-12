import json
import os

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, confusion_matrix

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.criteria_checker.save_features import load_feature_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list
from taskman_client.task_proxy import get_task_manager_proxy
import fire


def compute_f_k(y_true, y_pred, k = 9):
    m = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tp = m[1,1]
    fp = m[0,1]
    fn = m[1,0]

    prec = tp / (fp * k + tp)
    recall = tp / (tp + fn)
    if prec + recall == 0:
        return 0
    else:
        return 2 * prec * recall / (prec + recall)


def load_s9(dataset, run_name, seq, convert_score_fn) -> np.array:
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "j_res", dataset, f"{run_name}.json")
    j = json.load(open(res_save_path))
    X = []
    for data_id, _, ret_text, output in j:
        d = {}
        try:
            for code, term, score in output:
                prob = convert_score_fn(score, term)

                d[code] = prob

            x = []
            for item in seq:
                v = d[item] if item in d else 0
                x.append(v)
        except KeyError:
            print(output)
            raise

        X.append(x)

    return np.array(X)


def convert_score(score, term):
    prob = np.exp(score)
    if term == "Yes" or term == "yes":
        pass
    else:
        prob = -prob
    return prob


def convert_score_discrete(score, term):
    if term == "Yes" or term == "yes":
        return 1
    else:
        return 0


def main(sb):
    s9_run_name = "llama_s9_{}"
    s9_run_name = "llama_s9nosb"
    print("load_run_name_fmt", s9_run_name)
    f_name = "remindme"
    print("Use auxiliary feature", f_name)

    def load_here(dataset):
        X = load_s9(dataset, s9_run_name, target_seq, convert_score_fn)
        if f_name is None:
            x_i = [[0] for _ in range(len(X))]
        else:
            x_i = load_feature_csv(train_dataset, f_name)

        X = np.concatenate([X, x_i], axis=1)
        labels = load_labels(dataset)
        y = right(labels)
        return X, y

    target_seq = [f"S{i}" for i in range(1, 10)]
    proxy = get_task_manager_proxy()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    try:
        load_run_name = s9_run_name.format(sb)
        train_dataset = f"{sb}_2_train_100"
        val_dataset = f"{sb}_2_val_100"
        report_run_name = load_run_name + "_100d"
        convert_score_fn = convert_score_discrete
        X_train, y_train = load_here(train_dataset)
        X_test, y_test = load_here(val_dataset)
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)
        train_score = compute_f_k(y_train, train_pred)
        val_pred = clf.predict(X_test)
        val_score = compute_f_k(y_test, val_pred)
        print(f"\nResults for subreddit: {sb}")
        print(f"Training F-k Score: {train_score:.4f}")
        print(f"Validation F-k Score: {val_score:.4f}")
        print("coef", clf.coef_, clf.intercept_)
        # proxy.report_number(report_run_name, val_score, val_dataset, "f1")
    except FileNotFoundError as e:
        print(e)



if __name__ == "__main__":
    fire.Fire(main)




