import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.keyword_extractor import KeywordExtractor
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule2

import json


def main():
    extractor = KeywordExtractor()
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            raw_path = os.path.join(
                output_root_path, "reddit", "rule_processing", "keyword_raw", f"{sb}.json")
            make_parent_exists(raw_path)
            rules = load_reddit_rule2(sb)
            data = []
            for r in rules:
                rule_text = r["short_name"] + ". " + r["description"]
                keywords = extractor.extract_keywords(rule_text)
                data.append(keywords)
            json.dump(data, open(raw_path, "w"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()