import json

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.keyword_extractor import KeywordExtractor
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_path
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_auto_prompt_path



def split_clean_lines(rule_text):
    lines = rule_text.split("\n")
    lines = [line.strip() for line in lines]
    lines = [line for line in lines if line]
    lines = list(set(lines))
    return lines


def main():
    extractor = KeywordExtractor()
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            name = "chatgpt3"
            rule_text = open(get_reddit_auto_prompt_path("chatgpt3", sb), "r").read()
            lines = split_clean_lines(rule_text)

            save_path = get_named_keyword_path(name, sb)
            make_parent_exists(save_path)
            data = [extractor.extract_keywords(line) for line in lines]
            json.dump(data, open(save_path, "w"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
