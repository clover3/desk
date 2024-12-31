import os

import requests

from rule_gen.cpath import data_root_path
from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    sb_names = load_subreddit_list()
    for sb in sb_names:
        url = f"https://www.reddit.com/r/{sb}/"
        response = requests.get(url)
        html_save = os.path.join(
            data_root_path, "reddit", "pages", f"{sb}.html")
        open(html_save, "wb").write(response.content)

    return NotImplemented


if __name__ == "__main__":
    main()