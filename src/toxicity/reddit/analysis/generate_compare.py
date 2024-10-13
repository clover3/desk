from toxicity.reddit.compare import do_reddit_pred_compare
from toxicity.reddit.path_helper import get_group1_list


def main():
    subreddit_list = get_group1_list()
    for src in subreddit_list:
        run1 = f"bert_{src}"
        for dst in subreddit_list:
            run2 = f"bert_{dst}"
            for venu in subreddit_list:
                condition = f"{venu}_val_1K"
                do_reddit_pred_compare(condition, run1, run2)


if __name__ == "__main__":
    main()
