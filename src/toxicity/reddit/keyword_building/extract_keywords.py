import json
import os

from chair.misc_lib import make_parent_exists
from toxicity.apis.open_ai import OpenAIChatClient
from toxicity.cpath import output_root_path
from toxicity.reddit.dev_code.cola import save_data
from toxicity.reddit.path_helper import load_subreddit_list, load_reddit_rule2



def main():
    client = OpenAIChatClient("gpt-4o")
    inst = "Extract keywords from the following text. Return as a json list"
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
                prompt = inst + "\n <text>" + rule_text + "</text>"
                ret_text = client.request(prompt)
                data.append(ret_text)
                # print("==Prompt==")
                # print(prompt)
                # print("=====")
            json.dump(data, open(raw_path, "w"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()