import itertools
import json
import os
from collections import Counter

import tqdm
from nltk import ngrams

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv, save_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    sb_list = load_subreddit_list()
    save_path = os.path.join(output_root_path, "reddit", "near_duplicates.json")

    output = []
    n_item = 3000
    with open(save_path, "w") as f_out:
        for sb in tqdm.tqdm(sb_list):
            print("Processing", sb)
            try:
                data_name = "train_data2"
                items = read_csv(get_reddit_train_data_path_ex(
                    data_name, sb, "train"), return_itr=True)

                counter = Counter()
                for text, label_s in itertools.islice(items, n_item):
                    tokens = text.split()

                    for ten_gram in ngrams(tokens, 10):
                        key = tuple(ten_gram)
                        counter[key] += 1


                freq_keys = [key for key, cnt in counter.items() if cnt > 10]
                counter = Counter()
                for text, label_s in itertools.islice(items, n_item):
                    tokens = text.split()

                    for ten_gram in ngrams(tokens, 10):
                        key = tuple(ten_gram)
                        counter[key] += 1
                        if key in freq_keys:
                            break

                for key, cnt in counter.items():
                    if cnt > 10:
                        print(key, cnt)
                        f_out.write(json.dumps(key) + "\n")
                        output.append(key)

            except FileNotFoundError as e:
                print(e)


if __name__ == "__main__":
    main()
