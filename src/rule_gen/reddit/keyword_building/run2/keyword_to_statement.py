import json
import os

import fire

from chair.list_lib import flatten
from chair.misc_lib import make_parent_exists
from desk_util.open_ai import OpenAIChatClient
from rule_gen.reddit.keyword_building.apply_statement_common import statement_gen_prompt_fmt
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_path, \
    get_named_keyword_statement_path
from rule_gen.reddit.path_helper import load_subreddit_list


def main(sb=None):
    client = OpenAIChatClient("gpt-4o")
    sb_list = load_subreddit_list()
    if sb is not None:
        sb_list = [sb]
    name = "chatgpt3"
    for sb in sb_list:
        print(f"==== {sb} ====")
        try:
            save_path = get_named_keyword_path(name, sb)
            list_list = json.load(open(save_path, "r"))
            keywords = list(set(flatten(list_list)))
            keyword_statement_path = get_named_keyword_statement_path(name, sb)

            # if os.path.exists(keyword_statement_path):
            #     continue
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
    fire.Fire(main)
