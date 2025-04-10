import json
import os
import fire
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            path = os.path.join(output_root_path, "reddit", "rule_processing",
                                "cluster_probe_questions_dedup", f"bert2_{sb}.json")
            obj = json.load(open(path, "r"))
            json.dump(obj, open(path, "w"), indent=4, ensure_ascii=False)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
