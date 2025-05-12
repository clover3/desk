from collections import Counter

import fire

from desk_util.io_helper import read_csv
from rule_gen.reddit.criteria_checker.feature_valuation import feature_valuation_over_train_data2, \
    feature_valuation_inner
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex


def main(feature: str =""):
    def extract_feature(text):
        x_i = feature.lower() in text.lower()
        return x_i

    subreddit_list = get_split_subreddit_list("train")

    counter = Counter()
    for sb in subreddit_list:
        n_item = 3000
        data_name = "train_data2"
        items = read_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))
        items = items[:n_item]
        v = feature_valuation_inner(extract_feature, items, False)
        prob, cnt = v
        if cnt > 4 and prob > 0.8:
            counter["ban"] += 1
        if cnt > 4 and prob < 0.8:
            counter["no ban"] += 1
        else:
            counter["Not enough"] += 1

        print(sb, v)
    print(counter)


if __name__ == "__main__":
    fire.Fire(main)
