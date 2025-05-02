import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from rule_gen.reddit.bert_c.load_macro_norm_violation import load_all_norm_data_splits

def prepare_norm_dataset(split, id_mapping, n_policy):
    data = load_all_norm_data_splits()[split]
    all_data = []
    for norm_name, text_list in data:
        policy_label = [0] * n_policy
        norm_id = id_mapping[norm_name]
        policy_label[norm_id] = 1
        policy_label_mask = [1] * n_policy

        for t in text_list:
            e = {
                "text": t,
                "labels": 1,
                "policy_labels": policy_label,
                "policy_label_mask": policy_label_mask,
            }
            all_data.append(e)
    return all_data


def load_triplet_and_norm_dataset(triplet_path, norm_data, map_dict):
    norm_df = pd.DataFrame(norm_data)
    norm_df['sb_name'] = ['_unknown_'] * len(norm_data)
    norm_df['sb_id'] = norm_df['sb_name'].map(map_dict)
    n_policy = len(norm_df["policy_labels"][0])
    zeros = [0] * n_policy

    triplet_df = pd.read_csv(
        triplet_path,
        na_filter=False, keep_default_na=False,
        header=None, names=['sb_name', 'text', 'labels'],
        dtype={"sb_name": str, "text": str, 'labels': int})
    print("len(triplet_df)", len(triplet_df))
    triplet_df['sb_id'] = triplet_df['sb_name'].map(map_dict)
    triplet_df['policy_labels'] = [zeros] * len(triplet_df)
    triplet_df['policy_label_mask'] = [zeros] * len(triplet_df)
    print("True rate (triplet_df):", np.mean(triplet_df["labels"]))
    n_norm_records = int(len(triplet_df) * 0.2)
    norm_df = norm_df.sample(n_norm_records)
    print("Data size norm/ triplet df = {}/{}".format(len(norm_df), len(triplet_df)))
    combined_df = pd.concat([norm_df, triplet_df], ignore_index=True)

    print("True rate:", np.mean(combined_df["labels"]))
    combined_df = combined_df.sample(frac=1).reset_index(drop=True)
    combined_dataset = Dataset.from_pandas(combined_df)
    return combined_dataset
