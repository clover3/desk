from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def main():
    for split in ["train", "val", "test"]:
        train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", split)
        items = read_csv(train_data_path)
        valid = [t for t in items if len(t) == 2]
        save_csv(valid, get_reddit_train_data_path_ex("train_data2", "train_mix", split))


if __name__ == "__main__":
    main()
