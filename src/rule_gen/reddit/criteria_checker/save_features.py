import pandas as pd
import numpy as np
import tqdm
import fire

from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import save_csv, read_csv
from rule_gen.cpath import output_root_path
import os


def apply_feature(dataset, extract_feature):
    payload = load_csv_dataset_by_name(dataset)
    arr = []
    for data_id, text in tqdm.tqdm(payload):
        x_i = extract_feature(text)
        arr.append([x_i])
    arr = np.array(arr)
    return arr


def load_feature_csv(dataset, run_name):
    save_path: str = os.path.join(output_root_path, "features", f"{dataset}", f"{run_name}.csv")
    df = pd.read_csv(save_path, header=None)
    return df.values


def main(dataset, run_name, feature):
    def extract_feature(text):
        x_i = feature.lower() in text.lower()
        return int(x_i)

    arr = apply_feature(dataset, extract_feature)
    save_path: str = os.path.join(output_root_path, "features", f"{dataset}", f"{run_name}.csv")
    save_csv(arr, save_path)


if __name__ == "__main__":
    fire.Fire(main)
