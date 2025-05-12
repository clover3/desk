import json
import os
from collections import Counter

from chair.list_lib import right, left
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import load_csv_dataset
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex
from rule_gen.reddit.s9.feature_extractor import extract_ngram_features, get_value
from rule_gen.cpath import output_root_path
import os

def main():
    for split in ["train", "val"]:
        todo = get_split_subreddit_list(split)
        for sb in todo:
            out_save_path = os.path.join(
                output_root_path, "reddit",
                "rule_processing", "bot_trivial_features", f"{sb}.json")
            if not os.path.exists(out_save_path):
                continue
            items = json.load(open(out_save_path))
            data = read_csv(get_reddit_train_data_path_ex("train_data2", sb, "train"))
            data = data[:1000]
            text_list = left(data)
            for item in items:
                cnt = 0
                for text in text_list:
                    if item in text.lower():
                        cnt += 1
                print(item, cnt)

if __name__ == "__main__":
    main()