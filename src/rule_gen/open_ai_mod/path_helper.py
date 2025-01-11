import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path


def get_open_ai_mod_csv_path(role):
    return os.path.join(output_root_path, "open_ai_mod", f"{role}.csv")



def get_rule_gen_save_path(dataset_name: str, run_name: str) -> str:
    dir_name: str = f"{dataset_name}"
    file_name: str = f"{run_name}.json"
    save_path: str = os.path.join(output_root_path, "rule_gen", dir_name, file_name)
    make_parent_exists(save_path)
    return save_path

