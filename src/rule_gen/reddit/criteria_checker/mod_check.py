from krovetzstemmer import Stemmer
from nltk.tokenize import word_tokenize


import fire
from rule_gen.reddit.criteria_checker.feature_valuation import feature_valuation_over_train_data2
from rule_gen.reddit.path_helper import load_subreddit_list



def feature_check_w_stem(patterns, sb):
    stemmer = Stemmer()
    print(sb)
    # print("{} -> {}".format(feature, stemmer.stem(feature)))
    patterns = [stemmer(t) for t in patterns]

    def extract_feature(text):
        if "contact the moderators of this subreddit" in text:
            return 0

        tokens = word_tokenize(text)
        tokens = [stemmer(t) for t in tokens]
        for k in patterns:
            if k in tokens:
                return 1
        return 0

    n_item = 3000
    n_train = 2000
    feature_valuation_over_train_data2(extract_feature, n_item, n_train, sb)


def main():
    patterns = ["mod", "mods", "modded", "moderators", "moderator"]
    sb_list = load_subreddit_list()
    for sb in sb_list:
        feature_check_w_stem(patterns, sb)


if __name__ == "__main__":
    fire.Fire(main)
