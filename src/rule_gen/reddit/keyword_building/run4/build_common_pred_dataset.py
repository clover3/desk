import pickle
from collections import Counter
from rule_gen.cpath import output_root_path

import fire
import os
from chair.misc_lib import SuccessCounter
from chair.tab_print import print_table
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex
import numpy as np


def build_data(role = "train"):
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", role)
    items = read_csv(train_data_path)
    subreddit_list = get_split_subreddit_list("train")

    preds_d = {}
    not_available_list = []
    for sb in subreddit_list:
        try:
            run_name = "bert2_{}".format(sb)
            dataset = f"train_data2_mix_{role}"
            preds = load_clf_pred(dataset, run_name)
            if len(preds) == len(items):
                preds_d[sb] = preds
        except FileNotFoundError as e:
            not_available_list.append(sb)
            print(e)

    threshold = len(preds_d) * 0.75
    print("Not available: ", ", ".join(not_available_list))
    print("Available: ", ", ".join(preds_d.keys()))
    print("{} done".format(len(preds_d)))
    output = []
    counter = Counter()
    for idx in range(len(items)):
        cnt = 0
        for sb_k in preds_d:
            data_id, pred, _ = preds_d[sb_k][idx]
            if pred:
                cnt += 1
        text, label = items[idx]
        if cnt > threshold:
            new_label = 1
        else:
            new_label = 0
        counter[new_label] += 1
        output.append((text, new_label))

    print(counter)
    print("True rate", counter[1] / (counter[0] + counter[1]))
    save_path = get_reddit_train_data_path_ex(
        "train_data2", "train_mix_sel", role)
    save_csv(output, save_path)

if __name__ == "__main__":
    fire.Fire(build_data)
