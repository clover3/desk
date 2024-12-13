import json
import os

from chair.misc_lib import make_parent_exists
from toxicity.cpath import data_root_path, output_root_path
from toxicity.io_helper import read_csv_column, read_csv


def get_study_subreddit_list_path():
    save_path = os.path.join(data_root_path, "reddit", "study-subreddits.csv")
    return save_path


def get_split_subreddit_list_path(split):
    save_path = os.path.join(data_root_path, "reddit", f"subreddits_{split}.csv")
    return save_path


def get_split_subreddit_list(split):
    return read_csv_column(get_split_subreddit_list_path(split), 0)


def get_group1_list():
    save_path = os.path.join(output_root_path, "reddit", "group", f"group1.txt")
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



def get_reddit_db_path():
    return os.path.join(
        output_root_path, "reddit", "db")
