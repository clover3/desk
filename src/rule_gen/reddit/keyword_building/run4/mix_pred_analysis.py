import pickle
from collections import Counter
from rule_gen.cpath import output_root_path
import os
from chair.misc_lib import SuccessCounter
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex
import numpy as np


def pos_rate_per_clf():
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)
    subreddit_list = get_split_subreddit_list("train")
    sc = SuccessCounter()
    output = []
    for sb in subreddit_list:
        try:
            run_name = "bert2_{}".format(sb)
            dataset = "train_data2_mix"
            preds = load_clf_pred(dataset, run_name)
            for t in preds:
                sc.add(t[1])
            output.append((sb, sc.get_suc_prob()))
            if len(preds) != len(items):
                print("{} != {}".format(len(preds), len(items)))
                raise ValueError()
        except FileNotFoundError as e:
            raise e
    output.sort(key=lambda x: x[1], reverse=True)
    print_table(output)


def num_of_clf_per_text():
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)
    subreddit_list = get_split_subreddit_list("train")
    preds_d = {}
    not_available_list = []
    for sb in subreddit_list:
        try:
            run_name = "bert2_{}".format(sb)
            dataset = "train_data2_mix"
            preds = load_clf_pred(dataset, run_name)
            if len(preds) == len(items):
                preds_d[sb] = preds
        except FileNotFoundError as e:
            not_available_list.append(sb)
            print(e)

    print("Not available: ", ", ".join(not_available_list))
    print("Available: ", ", ".join(preds_d.keys()))
    print("{} done".format(len(preds_d)))
    counter = Counter()
    for idx in range(len(items)):
        cnt = 0
        for sb_k in preds_d:
            data_id, pred, _ = preds_d[sb_k][idx]
            if pred:
                cnt += 1
        counter[cnt] += 1

    acc = 0
    s = sum(counter.values())
    for i in range(max(counter.keys())+1):
        rate = counter[i] / s
        acc += rate
        print(i, "{0:.2f} {1:.2f}".format(rate, acc))



def save_as_np():
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)
    subreddit_list = get_split_subreddit_list("train")
    preds_d = {}
    not_available_list = []
    for sb in subreddit_list:
        try:
            run_name = "bert2_{}".format(sb)
            dataset = "train_data2_mix"
            preds = load_clf_pred(dataset, run_name)
            if len(preds) == len(items):
                preds_d[sb] = preds
        except FileNotFoundError as e:
            not_available_list.append(sb)
            print(e)

    keys = list(preds_d.keys())
    keys.sort()
    print("Not available: ", ", ".join(not_available_list))
    print("Available: ", ", ".join(preds_d.keys()))
    print("{} done".format(len(preds_d)))
    data = []
    for idx in range(len(items)):
        cnt = 0
        row = []
        for sb_k in keys:
            _, pred, _ = preds_d[sb_k][idx]
            row.append(pred)
        data.append(row)

    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    data = np.array(data)
    pickle.dump(data, open(feature_save_path, "wb"))


if __name__ == "__main__":
    save_as_np()
