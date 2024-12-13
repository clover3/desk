import json
import os
import random
from collections import Counter

import pyzstd

from chair.misc_lib import TELI
from toxicity.cpath import output_root_path, data_root_path
from toxicity.reddit.path_helper import load_subreddit_list


def main():
    subreddit_list: list[str] = load_subreddit_list()
    save_dir = os.path.join(output_root_path, "reddit", "subreddit_samples")
    counter = Counter()
    for sb in subreddit_list:
        path = os.path.join(save_dir, f"{sb}.jsonl")
        for line in open(path, "r"):
            j = json.loads(line)
            score = j["score"]
            if score > 1:
                key = "> 1"
            elif score == 1:
                key = "1"
            else:
                key = " <= 0"

            counter[key] += 1
    total = sum(counter.values())
    percentages = {k: round(v / total, 3) for k, v in counter.items()}
    print( percentages)


if __name__ == "__main__":
    main()