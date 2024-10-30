import requests
import os

from toxicity.cpath import output_root_path
from toxicity.reddit.crawl.rate_limiter import RateLimiter
from toxicity.reddit.path_helper import load_subreddit_list


def get_reddit_archive_save_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "wayback", f"{sb}.html")
    return rule_save_path


def download(url, save_path):
    response = requests.get(url)

    if response.status_code == 200:
        with open(save_path, "wb") as f:
            f.write(response.content)
    else:
        print(response)


def main():
    sb_names = load_subreddit_list()
    rate_limit = RateLimiter(15 - 5, 60)
    date_string = "20161225004811"
    date_string = "20161229083844"
    date_string = "20161109072457"
    for sb in sb_names[2:]:
        try:
            url = f"https://web.archive.org/web/{date_string}/https://www.reddit.com/r/{sb}/"
            save_path = get_reddit_archive_save_path(sb)
            if os.path.exists(save_path):
                continue
            print(sb)
            rate_limit.acquire()
            download(url, save_path)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
