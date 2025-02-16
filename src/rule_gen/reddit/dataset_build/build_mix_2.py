import os
import random
from collections import defaultdict

from desk_util.io_helper import read_csv, save_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, \
    get_reddit_train_data_path_ex


def main():
    train_data_root = os.path.join(output_root_path, "reddit", "train_data2")

    subreddit_list = get_split_subreddit_list("train")
    all_d = defaultdict(list)
    for subreddit in subreddit_list:
        for split in ["train", "val", "test"]:
            items = read_csv(os.path.join(train_data_root, subreddit, f"{split}.csv"))
            random.shuffle(items)
            all_d[split].extend(items[:1000])

    for split in ["train", "val", "test"]:
        items = all_d[split]
        random.shuffle(items)
        if split != "train":
            items = items[:1000]

        save_csv(items, get_reddit_train_data_path_ex("train_data2", "train_mix", split))


if __name__ == "__main__":
    main()
