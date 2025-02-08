import re
from collections import Counter

from desk_util.io_helper import read_csv
from rule_gen.reddit.dataset_build.common import generated_dataset_and_label
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_train_data_path_ex



def main():
    regex_list = ["a(ss|rse|es)([ -]?holes?)?", "b(i|\\*)o?(tch|\\*{3})(y|es)?", "cocks?([ -]?suck(ers?|ing)?)?",
     "cum(ming|[ -]shots?)?", "cunts?", "((mother|motha|mutha)[ -]?)?f(u?c?k?k|\\*ck|\\*{0,2}k|\\*{3})(er|ed|ing|s)?",
     "s(h(i|ar?|\\*)t|\\*{3}|h\\*{2})(s|ter|e|ting)?"]
    regex_list = [ "\\b[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\.[0-9]{1,3}\\b"]
    patterns = [re.compile(pattern, re.IGNORECASE) for pattern in regex_list]

    def return_match(text) -> str:
        for pattern in patterns:
            if pattern.search(text):
                return pattern.search(text).group()
        return ""  # Return empty string if no match is found

    subreddit_list = load_subreddit_list()
    split = "train"
    for subreddit in subreddit_list:
        data = read_csv(get_reddit_train_data_path_ex("train_data2", subreddit, split))
        counter = Counter()
        for text, label in data:
            ret = return_match(text)
            counter[bool(ret), int(label)] += 1

        print(subreddit, counter)
        for label in [0, 1]:
            # P(Label|True) = Count(True,Label) / Count(True,All)
            try:
                p_label_given_true = counter[True, label] / (counter[True, 0] + counter[True, 1])
            except ZeroDivisionError:
                p_label_given_true = float("nan")
            print(f"P(Label={label}|True) = {p_label_given_true:.3f} support={counter[True, label]}")


if __name__ == "__main__":
    main()