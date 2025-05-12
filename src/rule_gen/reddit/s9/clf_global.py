import json
import os
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from chair.list_lib import right
from chair.misc_lib import make_parent_exists
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.criteria_checker.save_features import load_feature_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.s9.classifier import convert_score, load_s9
from rule_gen.reddit.s9.clf_for_one import compute_f_k, convert_score_discrete


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
    X_train_all = []
    y_train_all = []
    for sb in sb_list:
        try:
            train_dataset = f"{sb}_2_train_100"
            convert_score_fn = convert_score
            X_train, y_train = load_here(train_dataset)
            X_train_all.append(X_train)
            y_train_all.append(np.array(y_train))
        except FileNotFoundError as e:
            print(e)


    X_train = np.concatenate(X_train_all, axis=0)
    y_train = np.concatenate(y_train_all, axis=0)
    clf = LogisticRegression(penalty="l1", solver="liblinear")
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)
    train_score = f1_score(y_train, train_pred)
    print(train_score)
    for sb in sb_list:
        try:
            load_run_name = s9_run_name.format(sb)
            val_dataset = f"{sb}_2_val_100"
            convert_score_fn = convert_score
            X_test, y_test = load_here(val_dataset)
            val_pred = clf.predict(X_test)
            val_score = f1_score(y_test, val_pred)
            f19 = compute_f_k(y_test, val_pred, 19)
            row = [sb, val_score, f19]
            table.append(row)
        except FileNotFoundError as e:
            print(e)

    table2 = []
    row2 = ["all", 0]
    for t in clf.coef_[0]:
        row2.append("{:.3f}".format(t))
    row2.append("{:.3f}".format(clf.intercept_[0]))
    table2.append(row2)

    print_table(table)
    print_table(table2)
    model_save_path = os.path.join(output_root_path, "models", "sklearn_run5", f"all.pickle")
    make_parent_exists(model_save_path)
    pickle.dump(clf, open(model_save_path, "wb"))


if __name__ == "__main__":
    main()




