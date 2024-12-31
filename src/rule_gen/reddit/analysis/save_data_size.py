from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.path_helper import get_reddit_training_data_size_path, get_reddit_train_data_path, load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    output = []
    for sb in subreddit_list:
        items = read_csv(get_reddit_train_data_path(sb, "train"))
        n = len(items)
        output.append((sb, n))

    save_csv(output, get_reddit_training_data_size_path())


if __name__ == "__main__":
    main()
