import os
from collections import Counter

from chair.misc_lib import get_second
from rule_gen.cpath import data_root_path
from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.path_helper import get_reddit_delete_post_path


def main():
    save_path = get_reddit_delete_post_path()
    all_data = read_csv(save_path)[1:]  # [body, subreddit]
    counter = Counter(map(get_second, all_data))
    save_path = os.path.join(data_root_path, "reddit", "reddit-removal-log-stat.csv")
    save_csv(counter.items(), save_path)


if __name__ == "__main__":
    main()
