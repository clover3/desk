
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


def get_popular_subreddit_names():
    file_name = "sfw_subreddits.jsonl"
    info_path = os.path.join(output_root_path, "reddit", "popular_list", file_name)
    j_itr = read_jsonl(info_path)
    return set(j["name"] for j in j_itr)




def main():
    # 1719806400
    # NoStupidQuestions is latest for being created at 2013-02-02 03:52:24
    # BaldursGate3 is latest for being created at 2019-05-30 10:35:38
    # Palworld is latest for being created at 2021-06-05 08:19:24
    # HonkaiStarRail is latest for being created at 2021-07-08 02:45:26
    # SipsTea is latest for being created at 2022-02-07 15:33:28
    # ChatGPT is latest for being created at 2022-12-01 15:19:27
    # AllThatIsInteresting is latest for being created at 2023-08-14 05:23:37
    # ChikaPH is latest for being created at 2023-10-02 05:02:05
    # SwiftlyNeutral is latest for being created at 2023-12-12 10:09:00
    # MildlyBadDrivers is latest for being created at 2024-01-27 12:55:09
    # infinitecraft is latest for being created at 2024-01-31 20:01:57

    reddit = praw.Reddit()
    start_sb = reddit.subreddit("PalWorldEngineering")
    june2024 = cut_line()
    print(june2024)
    latest = None
    params = {"after": start_sb.fullname}
    # params = {}
    cnt = 0
    # infinitecraft is latest for being created at 2024-01-31 20:01:57
    # popular returns 4388
    for sb in reddit.subreddits.popular(params=params, limit=8000):
        if latest is None:
            latest = sb.created_utc
        elif latest < sb.created_utc:
            datetime_object = datetime.datetime.fromtimestamp(sb.created_utc)
            print(f"{sb} is latest for being created at {datetime_object}")
            latest = sb.created_utc


        if sb.created_utc > june2024:
            print("After jun 2024")
            print(sb, sb.created_utc)

        cnt += 1
    print("cnt = ", cnt)



if __name__ == "__main__":
    main()