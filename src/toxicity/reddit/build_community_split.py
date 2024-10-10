import random

from toxicity.io_helper import save_csv, save_text_list_as_csv
from toxicity.path_helper import load_subreddit_list, get_split_subreddit_list_path


def main():
    sb_names = load_subreddit_list()
    random.shuffle(sb_names)

    parts = {"train": sb_names[:60],
             "val": sb_names[:80],
             "test": sb_names[:100]}
    for k, v in parts.items():
        p = get_split_subreddit_list_path(k)
        save_text_list_as_csv(v, p)





if __name__ == "__main__":
    main()