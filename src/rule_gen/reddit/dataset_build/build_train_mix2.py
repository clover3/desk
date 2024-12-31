import random

from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path, get_split_subreddit_list


def build_train():
    subreddit_list = get_split_subreddit_list("train")
    all = []
    for subreddit in subreddit_list:
        p = get_reddit_train_data_path(subreddit, "train")
        data = read_csv(p)
        random.shuffle(data)
        data = data[:10000]
        all.extend(data)

    random.shuffle(all)
    save_csv(all, get_reddit_train_data_path("train_mix2", "train"))


def build_val():
    num_item_per_sb_reddit = NotImplemented
    num_final_sel = NotImplemented
    
    subreddit_list = get_split_subreddit_list("train")
    all = []
    for subreddit in subreddit_list:
        p = get_reddit_train_data_path(subreddit, "val")
        data = read_csv(p)
        random.shuffle(all)
        data = data[:10000]
        all.extend(data)

    random.shuffle(all)
    all = all[:1000]
    save_csv(all, get_reddit_train_data_path("train_mix2", "val"))


if __name__ == "__main__":
    build_train()
    build_val()

