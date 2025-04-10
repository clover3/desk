
import json

from desk_util.io_helper import read_jsonl
from rule_gen.cpath import output_root_path
import os
import praw
import datetime
import time


import json
import os

from rule_gen.reddit.crawl.find_new_big_reddit.check_created_one_by_one import cut_line



def main():
    reddit = praw.Reddit()
    file_name = "new_sb_list.jsonl"
    start_sb_name = ""
    start_sb = reddit.subreddit(start_sb_name)
    print("Start from ", start_sb_name)
    params = {"after": start_sb.fullname}
    june2024 = cut_line()
    "infinitecraft"

    f = open(save_path, "a")
    finish = False
    while not finish:
        for i, sb in enumerate(reddit.subreddits.new(params=params, limit=1000)):
            datetime_object = datetime.datetime.fromtimestamp(sb.created_utc)

            save_j = {
                "name": sb.title,
                "id": sb.id,
                "created_utc": sb.created_utc,
            }
            if sb.created_utc < june2024:
                finish = True
            f.write(json.dumps(save_j) + '\n')
            f.flush()
        start_sb = sb
        params = {"after": start_sb.fullname}
        print(f"{sb} is created at {datetime_object}")
        print("Checked", i+1)

    print("Terminated")

if __name__ == "__main__":
    main()