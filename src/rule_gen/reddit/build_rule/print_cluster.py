import json
import os
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path


def main():
    run_name = "cluster"
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        print("Building prompt for", sb, "...")
        save_path = get_reddit_auto_prompt_path(run_name, sb)
        responses = json.load(open(save_path, "r"))

        for i in range(len(responses)):
            print("-----------------")
            print(responses[i])
        break


if __name__ == "__main__":
    main()
