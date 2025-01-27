import json
import os

from chair.misc_lib import make_parent_exists
from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import statement_gen_prompt_fmt
from rule_gen.reddit.keyword_building.path_helper import get_keyword_statement_path
from rule_gen.reddit.path_helper import enum_subreddit_w_rules


def main():
    client = OpenAIChatClient("gpt-4o")
    sb_list = enum_subreddit_w_rules()
    for sb in sb_list:
        print(f"==== {sb} ====")
        try:
            keywords = load_keywords(sb)
            keyword_statement_path = get_keyword_statement_path(sb)
            if os.path.exists(keyword_statement_path):
                continue

            make_parent_exists(keyword_statement_path)
            output = []
            for k in keywords:
                prompt = statement_gen_prompt_fmt.format(k)
                ret_text = client.request(prompt)
                output.append((k, ret_text))
                print(k, ret_text)

            json.dump(output, open(keyword_statement_path, "w"))
        except FileNotFoundError:
            pass




if __name__ == "__main__":
    main()
