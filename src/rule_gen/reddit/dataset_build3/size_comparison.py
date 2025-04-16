from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    two_names = ["train_data2", "train_data3"]
    head = [""] + two_names
    table = [head]
    for subreddit in subreddit_list:
        row = [subreddit]
        for name in two_names:
            file_path = os.path.join(
                output_root_path, "reddit",
                name, subreddit, "train.csv")
            l = len(read_csv(file_path))
            row.append(l)
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()