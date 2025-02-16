import os
import random
from collections import defaultdict, Counter

from desk_util.io_helper import read_csv, save_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, \
    get_reddit_train_data_path_ex


def main():
    train_data_root = os.path.join(output_root_path, "reddit", "train_data2")

    subreddit_list = get_split_subreddit_list("train")
    d = defaultdict(Counter)
    for subreddit in subreddit_list:
        print("Reading", subreddit)
        for split in ["train",]:
            items = read_csv(os.path.join(train_data_root, subreddit, f"{split}.csv"))

            for idx, pair in enumerate(items):
                try:
                    (text, label) = pair
                    d[text][(subreddit, label)] += 1
                except ValueError as e:
                    print(e)

    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)
    output = []
    for text, label in items:
        value = d[text]
        output.append((text, value))
    key_path = os.path.join(output_root_path, "reddit", "counter_train_data2_train_mix_key.csv")
    save_csv(output, key_path)


if __name__ == "__main__":
    main()
