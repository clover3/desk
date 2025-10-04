import os

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv_column
from rule_gen.cpath import data_root_path, output_root_path


def get_split_subreddit_list_path(split):
    save_path = os.path.join(data_root_path, "reddit", f"subreddits_{split}.csv")
    return save_path


def get_split_subreddit_list(split):
    if split == "both":
        return get_split_subreddit_list("train") + get_split_subreddit_list("val")
    else:
        return read_csv_column(get_split_subreddit_list_path(split), 0)


def get_reddit_delete_post_path():
    save_path = os.path.join(data_root_path, "reddit", "reddit-removal-log.csv")
    return save_path


def get_reddit_train_data_path_ex(data_name, sub_reddit, role):
    save_root = os.path.join(output_root_path, "reddit", data_name)
    save_dir = os.path.join(save_root, sub_reddit)
    save_path = os.path.join(save_dir, role + ".csv")
    return save_path


def get_rp_path(dir_name, file_name=None):
    if file_name is None:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name)
    else:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name, file_name)
    make_parent_exists(p)
    return p
