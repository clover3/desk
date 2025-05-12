import os
import pickle
from typing import Callable

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score

from chair.list_lib import right
from chair.misc_lib import make_parent_exists
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.s9.clf_for_one import compute_f_k
from rule_gen.reddit.s9.j_res_loader import load_s9


def run_exp_over_sb(sb_list, load_fn: Callable[[str, str], tuple]):
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    table = []
    for sb in sb_list:
        try:
            model_save_path = os.path.join(output_root_path, "models", "sklearn_run4", f"{sb}.pickle")
            make_parent_exists(model_save_path)
            X_train, y_train = load_fn(sb, "train")
            X_test, y_test = load_fn(sb, "val")
            clf = LogisticRegression(penalty="l1", solver="liblinear")

            clf.fit(X_train, y_train)
            clf.intercept_ = clf.intercept_ * 0
            train_pred = clf.predict(X_train)
            train_score = f1_score(y_train, train_pred)
            pickle.dump(clf, open(model_save_path, "wb"))
            val_pred = clf.predict(X_test)
            val_prec = precision_score(y_test, val_pred)
            val_recall = recall_score(y_test, val_pred)
            val_score = f1_score(y_test, val_pred)
            f19 = compute_f_k(y_test, val_pred, 19)
            row = [sb, train_score, val_score, f19, val_prec, val_recall]
            table.append(row)
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(sb)
            raise
    print_table(table)


def main():
    s9_run_name = "llama_s9nosb"
    print("load_run_name_fmt", s9_run_name)

    def load_fn(sb, split):
        dataset = f"{sb}_2_{split}_100"
        X_s9 = load_s9(dataset, s9_run_name)
        X = X_s9
        labels = load_labels(dataset)
        y = right(labels)
        return X, y

    run_exp_over_sb(get_split_subreddit_list("train"), load_fn)
    run_exp_over_sb(get_split_subreddit_list("val"), load_fn)


if __name__ == "__main__":
    main()
