from toxicity.reddit.compare import do_reddit_pred_compare
from toxicity.reddit.path_helper import get_group1_list, get_split_subreddit_list


def main():
    subreddit_list = get_split_subreddit_list("train")
    run1 = f"bert_train_mix3"
    for dst in subreddit_list:
        run2 = f"bert_{dst}"
        condition = f"{dst}_val_100"
        do_reddit_pred_compare(condition, run1, run2)


if __name__ == "__main__":
    main()
