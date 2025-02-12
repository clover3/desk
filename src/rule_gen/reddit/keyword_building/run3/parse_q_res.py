from json import JSONDecodeError

import numpy as np
import json
import os

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.keyword_extractor import parse_openai_json
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path


def load_parse(run_name, sb):
    save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                             f"{run_name}_questions_raw", f"bert2_{sb}.json")
    responses = json.load(open(save_path, "r", encoding="utf-8"))

    q_list: list[str] = []
    for r in responses:
        try:
            j: list[str] = parse_openai_json(r)
            q_list.extend(j)
        except JSONDecodeError as e:
            pass

    return q_list


def main():
    run_name = "cluster_probe"
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        print(f"==== {sb} ====")
        try:
            q_list = load_parse(run_name, sb)
            save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                     f"{run_name}_questions", f"bert2_{sb}.json")
            make_parent_exists(save_path)
            json.dump(q_list, open(save_path, "w"))

        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
