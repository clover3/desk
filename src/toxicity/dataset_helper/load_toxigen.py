import os
import random
from collections import Counter
from torch.utils.data import Dataset

from datasets import load_dataset
import logging

from rule_gen.cpath import output_root_path
from desk_util.csv_datasets import load_toxigen_para
from desk_util.io_helper import read_csv
from toxicity.llama_guard.llama_guard_formatter import LlamaGuardFormatter
from typing import List, Iterable, Tuple

_logger = logging.getLogger(__name__)


class ToxigenBinary:
    def __init__(self, split):
        self.split = split
        if split in ["train", "test"]:
            all_data = load_dataset("toxigen/toxigen-data", name="annotated")
            self.data = all_data[split]
        elif split == "train_fail100" or split == "train_fail100_para":
            save_path: str = os.path.join(output_root_path, "toxigen_fail", "train100_para.csv")
            print("loading from ", save_path)
            rows = read_csv(save_path)
            payload = []
            for data_id, label, ori_text, para_text in rows:
                if split == "train_fail100":
                    text = ori_text
                elif split == (""
                               ""):
                    text = para_text
                else:
                    raise ValueError
                e = {
                    "id": data_id,
                    "label": int(label),
                    "text": text,
                }
                payload.append(e)
            self.data = payload
        elif split == "1000_para":
            save_path: str = os.path.join(output_root_path, "toxigen_para", "toxigen1000_fail_para_res_man_fix.csv")
            rows = read_csv(save_path)
            payload = []
            for row in rows:
                data_id, text, label = row
                e = {
                    "id": data_id,
                    "label": int(label),
                    "text": text,
                }
                payload.append(e)
            self.data = payload

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.split in ["train", "test"]:
            label = 1 if item['toxicity_human'] > 3 else 0
            return {
                'id': str(idx),
                'text': item['text'],
                'label': label
            }
        else:
            return item


class ToxigenTrain(Dataset):
    def __init__(self):
        # There is no split
        all_data = load_dataset("toxigen/toxigen-data", name="train")["train"]
        data = list(all_data)
        random.seed(0)
        random.shuffle(data)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            'id': str(idx),
            'text': item['generation'],
            'label': item["prompt_label"],
        }


class FormattedToxigenDataset(Dataset):
    def __init__(self, base_dataset, formatter):
        self.base_dataset = base_dataset
        self.formatter = formatter

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        item = self.base_dataset[idx]
        item['text'] = self.formatter.get_prompt([item['text']])[0]
        return item

# Usage


def load_toxigen_formatted(split="train", n_item=None, subset="annotated") -> List[Tuple[str, str]]:
    data = load_toxigen_formatted_inner(split, subset)

    n_total = len(data)
    if n_item is not None:
        data = data[:n_item]

    _logger.info("%s split: Loaded %d items (of %d)", split, len(data), n_total)
    return data


def load_toxigen_formatted_inner(split, subset):
    # Step 1. Load toxigen dataset
    if subset == "annotated":
        ds = ToxigenBinary(split)
    elif subset == "train":
        ds = ToxigenTrain()
    else:
        raise ValueError(f"subset {subset} is not expected")
    return apply_llama_guard_formats(ds)


def apply_llama_guard_formats(ds: Iterable[dict]) -> list[tuple[str, str]]:
    formatter = LlamaGuardFormatter()
    data = []
    for e in ds:
        text = e['text']
        prompt = formatter.get_prompt([text])
        label = formatter.get_label_str(e['label'])
        data.append((prompt, label))
    return data


def load_toxigen_para_formatted():
    data = load_toxigen_para()
    return apply_llama_guard_formats(data)


if __name__ == '__main__':
    toxigen = ToxigenTrain()
    print(Counter([toxigen[i]["label"] for i in range(100)]))