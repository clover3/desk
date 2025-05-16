import itertools

import tqdm
from krovetzstemmer import Stemmer
from nltk import ngrams
from nltk.tokenize import word_tokenize

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv, save_csv, load_jsonl
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.path_helper import load_subreddit_list
from rule_gen.cpath import output_root_path
import os

def is_substring(a: list[str], b: list[str]) -> bool:
    if not a:
        return True  # An empty list is a substring of any list
    n, m = len(a), len(b)
    for i in range(m - n + 1):
        if b[i:i+n] == a:
            return True
    return False


def main():
    patterns = ["mod", "mods", "modded", "moderators", "moderator"]
    sb_list = load_subreddit_list()
    save_path = os.path.join(output_root_path, "reddit", "subset", "mod.csv")
    make_parent_exists(save_path)
    bot_pattern_path = os.path.join(output_root_path, "reddit", "bot_ngram.jsonl")
    bot_patterns = load_jsonl(bot_pattern_path)

    stemmer = Stemmer()
    patterns = [stemmer(t) for t in patterns]
    def is_bot_pattern(text):
        tokens = text.split()
        for k in bot_patterns:
            if is_substring(k, tokens):
                return True
        return False

    def extract_feature(text):
        any_match = False
        for k in patterns:
            if k in text.lower():
                any_match = True

        if not any_match:
            return 0
        tokens = word_tokenize(text)
        tokens = [stemmer(t) for t in tokens]
        for k in patterns:
            if k in tokens:
                return 1
        return 0

    # n_item = 3000
    all_entries = []
    for sb in tqdm.tqdm(sb_list):
        try:
            cnt = 0
            data_name = "train_data2"
            items = read_csv(get_reddit_train_data_path_ex(
                data_name, sb, "train"), return_itr=True)

            for text, label_s in items:
                x_i = extract_feature(text)
                if x_i and not is_bot_pattern(text):
                    row = [sb, text, label_s]
                    all_entries.append(row)
                    cnt += 1
                    if cnt >= 10:
                        break
        except FileNotFoundError as e:
            print(e)

    save_csv(all_entries, save_path)


if __name__ == "__main__":
    main()
