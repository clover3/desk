import json
import os

from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_rule_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    output = []
    columns = ["summary", "detail"]
    head = [""] + columns
    output.append(head)

    for sb in subreddit_list:
        rule_save_path = get_reddit_rule_path(sb)
        try:
            rules = json.load(open(rule_save_path, "r"))

            for r_idx, r in enumerate(rules):
                row = [sb, str(r_idx)]
                for c in columns:
                    row.append(rules[r_idx][c])
                output.append(row)
        except FileNotFoundError as e:
            pass

    save_path = os.path.join(output_root_path, "reddit", "train_rules.csv")
    save_csv(output, save_path)


if __name__ == "__main__":
    main()
