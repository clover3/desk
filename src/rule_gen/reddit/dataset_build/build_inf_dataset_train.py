from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_csv_dataset_path, get_label_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path, load_subreddit_list, get_reddit_train_data_path_ex


def main():
    subreddit_list = load_subreddit_list()
    todo = ["train"]
    todo = ["val", "test"]
    for subreddit in subreddit_list:
        for split in todo:
            data = read_csv(get_reddit_train_data_path_ex("train_data2", subreddit, split))
            dataset_name = f"{subreddit}_2_{split}"
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