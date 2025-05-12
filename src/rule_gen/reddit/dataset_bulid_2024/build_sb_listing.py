import csv
import os.path

from desk_util.path_helper import get_csv_dataset_path
from rule_gen.reddit.path_helper import load_subreddit_list, get_split_subreddit_list, \
    get_2024_split_subreddit_list_path


def main():
    for split in ["train", "val", "test"]:
        l = get_split_subreddit_list(split)
        valid_l = []
        for sb in l:
            dataset_name = "{}_2024b_100_test".format(sb)
            save_path = get_csv_dataset_path(dataset_name)
            if os.path.exists(save_path):
                valid_l.append(sb)

        save_path = get_2024_split_subreddit_list_path(split)
        with open(save_path, "w", newline='') as out_f:
            csv_writer = csv.writer(out_f)
            for sb in valid_l:
                csv_writer.writerow([sb])


if __name__ == "__main__":
    main()