import fire

from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_label_path
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        try:
            data_name = "{}_2024b_100_test".format(sb)
            labels = read_csv(get_label_path(data_name))

            new_labels = []
            for data_id, label in labels:
                new_label = 1 - int(label)
                new_labels.append((data_id, new_label))

            new_data_name = "{}_2024b_r_100_test".format(sb)
            save_csv(new_labels, get_label_path(new_data_name))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    fire.Fire(main)
