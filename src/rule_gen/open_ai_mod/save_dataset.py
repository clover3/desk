from datasets import load_dataset
import random

from desk_util.io_helper import save_csv
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


if __name__ == "__main__":
    save_open_ai_mod()