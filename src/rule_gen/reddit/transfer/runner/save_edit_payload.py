import os

from desk_util.io_helper import save_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, load_subreddit_list
from rule_gen.reddit.transfer.edit_exp import load_edit_payload


def main():
    subreddit_list = load_subreddit_list()
    save_root = os.path.join(output_root_path, "reddit", "train_head10.csv")
    all_data = []
    for subreddit in subreddit_list:
        data = load_edit_payload(subreddit)

        all_data.extend([(subreddit, a, b) for a, b in data])
    save_csv(all_data, save_root)


if __name__ == "__main__":
    main()
