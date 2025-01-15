import os

from datasets import load_dataset
import random

from desk_util.io_helper import save_csv, read_csv_column, read_csv
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_open_ai_mod_csv_path


def save_open_ai_mod():
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    data = []
    for item in openai_dataset:
        labels = {k: v for k, v in item.items() if k != "prompt"}
        any_true = any(labels.values())
        text = item["prompt"]
        data.append((text, int(any_true)))

    random.shuffle(data)
    train_len = int(len(data) * 0.8)
    val_len = int(len(data) * 0.1)

    save_path = get_open_ai_mod_csv_path("train")
    save_csv(data[:train_len], save_path)
    save_path = get_open_ai_mod_csv_path("val")
    save_csv(data[train_len:train_len + val_len], save_path)
    save_path = get_open_ai_mod_csv_path("test")
    save_csv(data[train_len + val_len:], save_path)


def save_to_dataset_dir():
    for split in ["train", "val", "test"]:
        save_path = get_open_ai_mod_csv_path(split)
        data = read_csv(save_path)
        name = "oam_{}".format(split)

        idx = 0
        dataset = []
        labels = []
        for text, label in data:
            data_id = "oam_{}_{}".format(split, idx)
            idx += 1
            dataset.append((data_id, text))
            labels.append((data_id, label))

        save_path: str = os.path.join(output_root_path, "datasets", f"{name}.csv")
        save_csv(dataset, save_path)
        save_path: str = os.path.join(output_root_path, "labels", f"{name}.csv")
        save_csv(labels, save_path)



def save_to_dataset_dir():
    abl_opt = ""
    k_list = []
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    data = []
    for item in openai_dataset:
        labels = {k: v for k, v in item.items() if k != "prompt"}
        any_true = any([labels[k] for k in k_list])
        text = item["prompt"]
        data.append((text, int(any_true)))

    random.shuffle(data)
    train_len = int(len(data) * 0.8)
    val_len = int(len(data) * 0.1)

    for split in ["train", "val", "test"]:
        data = NotImplemented
        name = "oam_{}_{}".format(abl_opt, split)

        idx = 0
        dataset = []
        labels = []
        for text, label in data:
            data_id = "oam_{}_{}".format(split, idx)
            idx += 1
            dataset.append((data_id, text))
            labels.append((data_id, label))

        save_path: str = os.path.join(output_root_path, "datasets", f"{name}.csv")
        save_csv(dataset, save_path)
        save_path: str = os.path.join(output_root_path, "labels", f"{name}.csv")
        save_csv(labels, save_path)


if __name__ == "__main__":
    save_to_dataset_dir()
