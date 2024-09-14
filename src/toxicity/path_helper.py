
import os

from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv
from chair.misc_lib import make_parent_exists


def get_dataset_pred_save_path(run_name: str, dataset_name: str) -> str:
    dir_name: str = f"{dataset_name}"
    file_name: str = f"{run_name}.csv"
    save_path: str = os.path.join(output_root_path, "gen_out", dir_name, file_name)
    make_parent_exists(save_path)
    return save_path


def get_label_path(dataset_name: str) -> str:
    file_name: str = f"{dataset_name}.csv"
    save_path: str = os.path.join(output_root_path, "labels", file_name)
    make_parent_exists(save_path)
    return save_path


def get_comparison_save_path(run_name: str, dataset_name: str) -> str:
    dir_name: str = f"{dataset_name}"
    file_name: str = f"{run_name}.csv"
    save_path: str = os.path.join(output_root_path, "comparison", dir_name, file_name)
    make_parent_exists(save_path)
    return save_path



def get_toxigen_failure_save_path(dataset_name, run_name):
    file_name: str = f"{dataset_name}_{run_name}.csv"
    save_path: str = os.path.join(output_root_path, "toxigen_fail", file_name)
    return save_path


def get_text_list_save_path(dataset_name):
    file_name: str = f"{dataset_name}.csv"
    save_path: str = os.path.join(output_root_path, "text_list", file_name)
    make_parent_exists(save_path)
    return save_path


def load_csv_dataset(dataset):
    save_path = get_csv_dataset_path(dataset)
    payload = read_csv(save_path)
    return payload


def get_csv_dataset_path(dataset):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    return save_path

