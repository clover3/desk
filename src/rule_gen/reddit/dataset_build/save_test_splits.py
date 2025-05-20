from desk_util.io_helper import save_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_subreddit_list, get_split_subreddit_list_path


def main():
    train_val = get_split_subreddit_list("both")
    all_sb = load_subreddit_list()
    test_sb = [sb for sb in all_sb if sb not in train_val]

    save_path = get_split_subreddit_list_path("test")
    open(save_path, "w").write("\n".join(test_sb))


if __name__ == "__main__":
    main()