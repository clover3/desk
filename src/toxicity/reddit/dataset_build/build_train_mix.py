import random

from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_csv_dataset_path, get_label_path
from toxicity.reddit.path_helper import get_reddit_train_data_path, load_subreddit_list, get_split_subreddit_list


def build_train():
    subreddit_list = get_split_subreddit_list("train")
    all = []
    for subreddit in subreddit_list:
        p = get_reddit_train_data_path(subreddit, "train")
        data = read_csv(p)
        all.extend(data)

    random.shuffle(all)
    save_csv(all, get_reddit_train_data_path("train_mix", "train"))


def build_val():
    subreddit_list = get_split_subreddit_list("train")
    all = []
    for subreddit in subreddit_list:
        p = get_reddit_train_data_path(subreddit, "val")
        data = read_csv(p)
        all.extend(data)

    random.shuffle(all)
    all = all[:1000]
    save_csv(all, get_reddit_train_data_path("train_mix", "val"))


if __name__ == "__main__":
    build_train()
    build_val()

