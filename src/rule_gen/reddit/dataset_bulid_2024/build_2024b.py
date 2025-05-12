import random

from desk_util.io_helper import load_jsonl
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.dataset_build.common import generated_dataset_and_label
from rule_gen.reddit.path_helper import load_subreddit_list, get_n_rules

def load_for(sb):
    dir_path = os.path.join(output_root_path, "reddit", "2024data")
    out_d = {}
    for label in ["pos", "neg"]:
        save_path = os.path.join(dir_path, f"{sb}-{label}.jsonl")
        j_list = load_jsonl(save_path)
        out_d[label] = [j["text"] for j in j_list]
    return out_d


def main():
    sb_list = load_subreddit_list()
    random.seed(42)
    n_item = 50
    for sb in sb_list:
        try:
            out_d = load_for(sb)
            pos_list: list[str] = out_d["pos"]
            neg_list: list[str] = out_d["neg"]
            random.shuffle(neg_list)
            neg_list = neg_list[:len(pos_list)]

            if len(pos_list) < n_item:
                print("Skip", sb)

            pos_list = pos_list[:n_item]
            neg_list = neg_list[:n_item]

            data = []
            for t in pos_list:
                data.append((t, 1))
            for t in neg_list:
                data.append((t, 0))
            random.shuffle(data)

            dataset_name = f"{sb}_2024b_100_test"
            print(sb, len(pos_list), len(neg_list))
            generated_dataset_and_label(data, dataset_name)
        except FileNotFoundError:
            pass

if __name__ == '__main__':
    main()

