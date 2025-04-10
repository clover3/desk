import json

from desk_util.io_helper import read_jsonl
from rule_gen.cpath import output_root_path
import os
import praw
import datetime
import time


def cut_line():
    # Specify your date
    dt = datetime.datetime(
        2024, 7, 1, 0, 0, 0)  # year, month, day, hour, minute, second
    # Convert to Unix timestamp
    timestamp = int(time.mktime(dt.timetuple()))
    return timestamp


def main():
    reddit = praw.Reddit()
    def get_created_utc(name):
        sb = reddit.subreddit(name)
        return sb.created_utc

    june2024 = cut_line()

    file_name = "sfw_subreddits.jsonl"
    info_path = os.path.join(output_root_path, "reddit", "popular_list", file_name)
    j_itr = read_jsonl(info_path)

    file_name = "sfw_subreddits_created.jsonl"
    save_path = os.path.join(output_root_path, "reddit", "popular_list", file_name)

    f = open(save_path, "w")
    for i, j in enumerate(j_itr):
        j['created'] = get_created_utc(j['name'])
        if j['created'] > june2024:
            print(j['name'])
        f.write(json.dumps(j) + '\n')
        if i % 100 == 0:
            print(i)

    f.close()



if __name__ == "__main__":
    main()