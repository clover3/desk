from toxicity.io_helper import read_csv
from toxicity.reddit.path_helper import get_reddit_delete_post_path


def main():
    data = read_csv(get_reddit_delete_post_path())
    print(len(data))


if __name__ == "__main__":
    main()