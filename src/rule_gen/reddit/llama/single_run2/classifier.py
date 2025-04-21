import numpy as np
import json

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.path_helper import get_split_subreddit_list
from taskman_client.task_proxy import get_task_manager_proxy


def load(dataset, run_name, seq, convert_score_fn):
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "gpt_res", dataset, f"{run_name}.json")
    j = json.load(open(res_save_path))
    labels = load_labels(dataset)

    y = right(labels)
    X = []
    for data_id, _, ret_text, output in j:
        d = {}
        for code, term, score in output:
            prob = convert_score_fn(score, term)

            d[code] = prob
        try:
            x = [d[item] for item in seq]
        except KeyError:
            x = None
        X.append(x)

    items = [(x_i, y_i) for x_i, y_i in zip(X, y) if x_i is not None]
    x_itr, y_itr = zip(*items)
    return list(x_itr), list(y_itr)


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


def main():
    table = []
    # sb_list = get_split_subreddit_list("train")
    sb_list = get_split_subreddit_list("val")

    target_seq = [f"S{i}" for i in range(1, 10)]
    proxy = get_task_manager_proxy()
    np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})
    table2 = []


    for sb in sb_list:
        try:
            load_run_name = f"llama_s9_{sb}"
            report_run_name = load_run_name + "_100d"
            convert_score_fn = convert_score_discrete
            X_train, y_train = load(f"{sb}_2_train_100", load_run_name, target_seq,
                                    convert_score_fn
                                    )
            val_dataset = f"{sb}_2_val_100"
            X_test, y_test = load(
                val_dataset, load_run_name, target_seq, convert_score_fn)

            # print("loaded {}/{} for {}".format(len(X_train), len(X_test), sb))
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)

            train_score = f1_score(y_train, train_pred)
            val_pred = clf.predict(X_test)
            val_score = f1_score(y_test, val_pred)
            row = [sb, val_score]
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




