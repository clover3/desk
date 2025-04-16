from desk_util.io_helper import read_csv
from rule_gen.reddit.dataset_build.common import generated_dataset_and_label
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_train_data_path_ex


def main():
    subreddit_list = load_subreddit_list()
    todo = ["train", "val", "test"]
    for subreddit in subreddit_list:
        for split in todo:
            data = read_csv(get_reddit_train_data_path_ex("train_data3", subreddit, split))
            dataset_name = f"{subreddit}_3_{split}"
            generated_dataset_and_label(data, dataset_name)


if __name__ == "__main__":
    main()
