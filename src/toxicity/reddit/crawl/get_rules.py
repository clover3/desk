import json

import praw
import prawcore

from toxicity.reddit.path_helper import get_reddit_rule_path2, load_subreddit_list


def main():
    sb_list = load_subreddit_list()
    reddit = praw.Reddit()
    for sb in sb_list:
        try:
            subreddit = reddit.subreddit(sb)
            rules = subreddit.rules()["rules"]
            save_path = get_reddit_rule_path2(sb)
            json.dump(rules, open(save_path, "w"))
        except prawcore.exceptions.NotFound as e:
            print(sb, e)


if __name__ == "__main__":
    main()