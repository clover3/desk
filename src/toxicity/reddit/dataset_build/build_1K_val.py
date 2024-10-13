from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_csv_dataset_path, get_label_path
from toxicity.reddit.path_helper import get_reddit_train_data_path, load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    for subreddit in subreddit_list:
        for split in ["val", "test"]:
            dataset_name = f"{subreddit}_{split}"
            new_dataset_name = f"{subreddit}_{split}_1K"

            data = read_csv(get_csv_dataset_path(dataset_name))
            save_csv(data[:1000], get_csv_dataset_path(new_dataset_name))

            data = read_csv(get_label_path(dataset_name))
            save_csv(data[:1000], get_label_path(new_dataset_name))




def main100():
    subreddit_list = load_subreddit_list()
    for subreddit in subreddit_list:
        for split in ["val", "test"]:
            dataset_name = f"{subreddit}_{split}"
            new_dataset_name = f"{subreddit}_{split}_100"

            data = read_csv(get_csv_dataset_path(dataset_name))
            save_csv(data[:100], get_csv_dataset_path(new_dataset_name))

            data = read_csv(get_label_path(dataset_name))
            save_csv(data[:100], get_label_path(new_dataset_name))



if __name__ == "__main__":
    main100()

