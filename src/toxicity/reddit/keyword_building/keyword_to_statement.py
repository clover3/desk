import json
import os

from chair.misc_lib import make_parent_exists
from toxicity.apis.open_ai import OpenAIChatClient
from toxicity.cpath import output_root_path
from toxicity.reddit.keyword_building.inf_keyword_to_text import load_keywords
from toxicity.reddit.path_helper import load_subreddit_list, load_reddit_rule2, enum_subreddit_w_rules


def main():
    prompt_fmt = """
keyword: {}
With the keyword above, write a statement like:
* This text contain A.
* This text is A.
* This text is considered A.

Only output a single statement. 
"""
    client = OpenAIChatClient("gpt-4o")
    sb_list = enum_subreddit_w_rules()
    for sb in sb_list:
        print(f"==== {sb} ====")
        try:
            keywords = load_keywords(sb)
            keyword_statement_path = os.path.join(
                output_root_path, "reddit", "rule_processing",
                "keyword_statement", f"{sb}.json")
            if os.path.exists(keyword_statement_path):
                continue

            make_parent_exists(keyword_statement_path)
            output = []
            for k in keywords:
                prompt = prompt_fmt.format(k)
                ret_text = client.request(prompt)
                output.append((k, ret_text))
                print(k, ret_text)

            json.dump(output, open(keyword_statement_path, "w"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
