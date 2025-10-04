import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path


def get_model_save_path(name):
    save_path = os.path.join(output_root_path, "models", name)
    make_parent_exists(save_path)
    return save_path


def get_model_log_save_dir_path(name):
    save_path = os.path.join(output_root_path, "models", name, "log")
    make_parent_exists(save_path)
    return save_path
