from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_csv_dataset_path, get_label_path
from toxicity.reddit.path_helper import get_reddit_train_data_path, load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    for subreddit in subreddit_list:
        for split in ["val", "test"]:
            data = read_csv(get_reddit_train_data_path(subreddit, split))
            dataset_name = f"{subreddit}_{split}"
            data_w_id = []
            for idx, (text, label) in enumerate(data):
                data_id = f"{dataset_name}_{idx}"
                data_w_id.append((data_id, text, label))

            payload = [(e[0], e[1]) for e in data_w_id]
            labels = [(e[0], e[2]) for e in data_w_id]
            save_path = get_csv_dataset_path(dataset_name)
            save_csv(payload, save_path)
            save_path = get_label_path(dataset_name)
            save_csv(labels, save_path)



if __name__ == "__main__":
    main()