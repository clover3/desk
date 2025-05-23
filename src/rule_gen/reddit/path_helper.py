import json
import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import data_root_path, output_root_path
from desk_util.io_helper import read_csv_column, read_csv


def get_study_subreddit_list_path():
    save_path = os.path.join(data_root_path, "reddit", "study-subreddits.csv")
    return save_path


def get_split_subreddit_list_path(split):
    save_path = os.path.join(data_root_path, "reddit", f"subreddits_{split}.csv")
    return save_path


def get_2024_split_subreddit_list_path(split):
    save_path = os.path.join(data_root_path, "reddit", f"subreddits_2024_{split}.csv")
    return save_path

def load_2024_split_subreddit_list(split):
    if split == "both":
        return load_2024_split_subreddit_list("train") + load_2024_split_subreddit_list("val")
    else:
        return read_csv_column(get_2024_split_subreddit_list_path(split), 0)


def get_split_subreddit_list(split):
    if split == "both":
        return get_split_subreddit_list("train") + get_split_subreddit_list("val")
    else:
        return read_csv_column(get_split_subreddit_list_path(split), 0)


def get_split_display_list(split):
    all_items = get_split_subreddit_list(split)
    all_items.sort(key=lambda item: item.lower())
    all_items = [t for t in all_items if t not in ["Incels", "soccerstreams"]]
    return all_items


def get_group1_list():
    save_path = os.path.join(output_root_path, "reddit", "group", f"group1.txt")
    return read_csv_column(save_path, 0)


def get_group2_list():
    save_path = os.path.join(output_root_path, "reddit", "group", f"group2.txt")
    return read_csv_column(save_path, 0)


def get_reddit_delete_post_path():
    save_path = os.path.join(data_root_path, "reddit", "reddit-removal-log.csv")
    return save_path


def get_reddit_training_data_size_path():
    save_path = os.path.join(output_root_path, "reddit", "train_data_size.csv")
    return save_path


def get_reddit_training_data_size():
    save_path = get_reddit_training_data_size_path()
    return {k: int(v) for k, v in read_csv(save_path)}


def get_reddit_train_data_path(sub_reddit, role):
    save_root = os.path.join(output_root_path, "reddit", "train_data")
    save_dir = os.path.join(save_root, sub_reddit)
    save_path = os.path.join(save_dir, role + ".csv")
    return save_path


def get_reddit_train_data_path_ex(data_name, sub_reddit, role):
    save_root = os.path.join(output_root_path, "reddit", data_name)
    save_dir = os.path.join(save_root, sub_reddit)
    save_path = os.path.join(save_dir, role + ".csv")
    return save_path


def load_subreddit_list():
    save_path = get_study_subreddit_list_path()
    return read_csv_column(save_path, 0)


def enum_subreddit_w_rules():
    save_path = get_study_subreddit_list_path()
    itr = read_csv_column(save_path, 0)
    for sb in itr:
        if os.path.exists(get_reddit_rule_path2(sb)):
            yield sb


def get_reddit_manual_prompt_path(name):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "man_prompt", f"{name}.txt")
    return rule_save_path



def get_reddit_auto_prompt_path(type_name, name):
    rule_save_path = os.path.join(
        output_root_path, "reddit", f"{type_name}_prompt", f"{name}.txt")
    make_parent_exists(rule_save_path)
    return rule_save_path


def get_reddit_rule_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules", f"{sb}.json")
    return rule_save_path


def get_reddit_rule_path2(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules2", f"{sb}.json")
    make_parent_exists(rule_save_path)
    return rule_save_path



def load_reddit_rule(sb):
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    return rules



def load_reddit_rule2(sb):
    rule_save_path = get_reddit_rule_path2(sb)
    rules = json.load(open(rule_save_path, "r"))
    return rules


def load_reddit_rule_para(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rules_para", f"{sb}.txt")
    rule_text = open(rule_save_path, "r").read()
    return rule_text



def get_reddit_db_dir_path():
    return os.path.join(
        output_root_path, "reddit", "db")


def get_n_rules(sb):
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    n_rule = len(rules)
    return n_rule


def get_rp_path(dir_name, file_name=None):
    if file_name is None:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name)
    else:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name, file_name)
    make_parent_exists(p)
    return p


def get_j_res_save_path(run_name, dataset):
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "j_res", dataset, f"{run_name}.json")
    return res_save_path


def load_j_res(run_name, dataset):
    return json.load(open(get_j_res_save_path(run_name, dataset), "r"))