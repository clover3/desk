"""
Updated path_helper.py with HF support.
Always downloads from HuggingFace if files don't exist locally.
"""
import os
from desk_util.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv_column
from rule_gen.cpath import data_root_path, output_root_path
from rule_gen.hf_dataset_loader import get_hf_manager


def get_split_subreddit_list_path(split):
    save_path = os.path.join(data_root_path, "reddit", f"subreddits_{split}.csv")

    # Download from HF if file doesn't exist
    if not os.path.exists(save_path):
        hf_path = f"reddit/subreddits_{split}.csv"
        return get_hf_manager().get_csv_path(hf_path)

    return save_path


def get_split_subreddit_list(split):
    if split == "both":
        return get_split_subreddit_list("train") + get_split_subreddit_list("val")
    else:
        return read_csv_column(get_split_subreddit_list_path(split), 0)


def get_reddit_delete_post_path():
    save_path = os.path.join(data_root_path, "reddit", "reddit-removal-log.csv")

    if not os.path.exists(save_path):
        hf_path = "reddit/reddit-removal-log.csv"
        return get_hf_manager().get_csv_path(hf_path)

    return save_path


def get_reddit_train_data_path_ex(data_name, sub_reddit, role):
    save_root = os.path.join(output_root_path, "reddit", data_name)
    save_dir = os.path.join(save_root, sub_reddit)
    save_path = os.path.join(save_dir, role + ".csv")

    if not os.path.exists(save_path):
        hf_path = f"reddit/{data_name}/{sub_reddit}/{role}.csv"
        return get_hf_manager().get_csv_path(hf_path)

    return save_path


def get_rp_path(dir_name, file_name=None):
    if file_name is None:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name)
    else:
        p = os.path.join(output_root_path, "reddit", "rule_processing", dir_name, file_name)

        # Download intermediate outputs from HF if they don't exist
        if not os.path.exists(p) and file_name:
            hf_path = f"reddit/rule_processing/{dir_name}/{file_name}"
            try:
                return get_hf_manager().get_pickle_path(hf_path)
            except:
                # If not found in HF, create the path locally
                pass

    make_parent_exists(p)
    return p


def get_model_save_path(name):
    """
    Updated to support downloading models from HF.
    """
    save_path = os.path.join(output_root_path, "models", name)

    # Check if model exists locally, otherwise download from HF
    if not os.path.exists(os.path.join(save_path, "config.json")):
        try:
            return get_hf_manager().get_model_path(name)
        except:
            # Model not in HF, use local path
            pass

    make_parent_exists(save_path)
    return save_path


def get_model_log_save_dir_path(name):
    save_path = os.path.join(output_root_path, "models", name, "log")
    make_parent_exists(save_path)
    return save_path