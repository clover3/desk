import os

from chair.tab_print import tab_print
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list_path, get_split_subreddit_list


def main():
    sb_list = get_split_subreddit_list("train")
    for sb in sb_list:
        entail_save_path = os.path.join(output_root_path, "reddit",
                                        "rule_processing", "k_chatgpt3_to_text_100", f"{sb}.csv")
        f_train = os.path.exists(entail_save_path)
        entail_save_path = os.path.join(
            output_root_path, "reddit",
            "rule_processing", "k_chatgpt3_to_text_val_100", f"{sb}.csv")
        f_val = os.path.exists(entail_save_path)

        row = [sb, f_train, f_val]
        tab_print(row)


if __name__ == "__main__":
    main()
