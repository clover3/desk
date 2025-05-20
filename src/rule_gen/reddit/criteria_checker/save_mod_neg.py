import json
import os
import random

from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path


def main():
    text_path = os.path.join(output_root_path, "reddit", "subset", "mod.csv")
    data = read_csv(text_path)

    text_list = []
    for sb, text, label_s in data:
        if "thanks for submitting" in text:
            continue
        if "www.reddit.com" in text:
            continue
        if int(label_s[0]) == 0:
            text_list.append(text)

    random.shuffle(text_list)
    label_name = "neg"
    neg_path = os.path.join(output_root_path, "reddit", "subset", f"mod_{label_name}.csv")
    json.dump(text_list[:70], open(neg_path, "w"), indent=4)


if __name__ == "__main__":
    main()
