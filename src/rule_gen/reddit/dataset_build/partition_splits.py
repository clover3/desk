from desk_util.io_helper import save_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_subreddit_list, get_split_subreddit_list_path


def main():
    train_list = get_split_subreddit_list("train")


    def save_list_as(l, name):
        save_path = get_split_subreddit_list_path(name)
        open(save_path, "w").write("\n".join(l))

    save_list_as(train_list[:20], "sp1")
    save_list_as(train_list[20:40], "sp2")
    save_list_as(train_list[40:], "sp3")


if __name__ == "__main__":
    main()