import csv
import os
import random
from sklearn.model_selection import train_test_split

from chair.misc_lib import group_by, get_second
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv, read_jsonl, save_csv
from rule_gen.reddit.path_helper import get_reddit_delete_post_path
import re


def preprocess_text(text_list):
    text_list = [text.replace('\n', ' ') for text in text_list]
    text_list = [text.lower() for text in text_list]
    text_list = [" ".join(re.findall(r'[\w]+', text)) for text in text_list]
    return text_list

# Added filter conditions

def load_deletion_rate() -> dict[str, float]:
    file_path = os.path.join(output_root_path, "reddit", "delete_rate_2024.tsv")
    rate_d = {}
    reader = csv.reader(open(file_path, "r", newline=''), delimiter="\t")
    for sb, _, _, rate_s in reader:
        rate_d[sb] = float(rate_s)
    return rate_d


def main():
    save_path = get_reddit_delete_post_path()
    all_data = read_csv(save_path)[1:]  # [body, subreddit]
    grouped = group_by(all_data, get_second)
    all_del_rate = 0.037
    load_dir = os.path.join(output_root_path, "reddit", "subreddit_samples")
    save_root = os.path.join(output_root_path, "reddit", "train_data3")
    del_rate_d: dict[str, float] = load_deletion_rate()
    n_pos_max = 10
    for sub_reddit in grouped:
        try:
            del_rate: float = del_rate_d[sub_reddit]
            if del_rate == 0:
                del_rate = all_del_rate
        except KeyError:
            del_rate = all_del_rate
        print(sub_reddit)
        pos_to_neg_rate = del_rate / (1 - del_rate)

        try:
            file_path = os.path.join(load_dir, sub_reddit + ".jsonl")
            pos_items = grouped[sub_reddit]
            pos_texts = [t[0] for t in pos_items]
            pos_texts_set = set(pos_texts)
            print(f"{len(pos_items)} pos_items")

            def exclude(j):
                text = j["body"]
                return text == "[removed]" or text == "[deleted]" or text in pos_texts_set or j["score"] < 1

            neg_items = read_jsonl(file_path)
            n_neg_before = len(neg_items)
            neg_items = [e for e in neg_items if not exclude(e)]
            n_neg_after = len(neg_items)
            print(f"{len(neg_items)} neg_items ({n_neg_before - n_neg_after} removed)")

            if len(pos_items) < len(neg_items) * pos_to_neg_rate:
                # Match negative items to pos_items
                n_pos_use = min(len(pos_items), n_pos_max)
                n_neg_use = n_pos_use * (n_pos_use / pos_to_neg_rate)
                n_neg_use = int(n_neg_use)
            else:
                # Match positive items to neg_items
                n_pos_use = len(neg_items) * (del_rate / (1 - del_rate))
                n_pos_use = int(n_pos_use)
                n_neg_use = len(neg_items)

            print("Used pos/neg = {}/{}".format(n_pos_use, n_neg_use))
            pos_texts = pos_texts[:n_pos_use]
            neg_items = neg_items[:n_neg_use]

            neg_texts = [j['body'] for j in neg_items]
            neg_pairs = [(t, 0) for t in neg_texts]
            pos_pairs = [(t, 1) for t in pos_texts]
            all_pairs = pos_pairs + neg_pairs
            random.shuffle(all_pairs)
            train_data, val_test_data = train_test_split(all_pairs, test_size=0.1, random_state=42)
            val_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)
            todo = [
                ("train", train_data),
                ("val", val_data),
                ("test", test_data),
            ]
            save_dir = os.path.join(save_root, sub_reddit)
            for role, data in todo:
                save_path = os.path.join(save_dir, role + ".csv")
                save_csv(data, save_path)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
