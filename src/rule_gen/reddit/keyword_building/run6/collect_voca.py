import json
import os
import pickle
from collections import defaultdict

from nltk import ngrams

from rule_gen.reddit.keyword_building.run6.common import get_bert_basic_tokenizer
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    voca = defaultdict(set)
    tokenize_fn = get_bert_basic_tokenizer()
    for sb in subreddit_list:
        print(sb)
        save_path = get_rp_path("s9_ngram_93", f"{sb}.json")
        if not os.path.exists(save_path):
            continue

        items = json.load(open(save_path))
        for res in items:
            for k in res["strong_sub_texts"]:
                sub_text: str = k["sub_text"]
                tokens = tokenize_fn(sub_text)
                for t in tokens:
                    voca[1].add(t)

                n = 2
                while n <= len(tokens) and n <= 30:
                    for seq in ngrams(tokens, n):
                        voca[n].add(tuple(seq))
                    n += 1

    save_path = get_rp_path("run6_voca.pkl")
    pickle.dump(voca, open(save_path, "wb"))


def show_cnt():
    save_path = get_rp_path("run6_voca.pkl")
    voca = pickle.load(open(save_path, "rb"))

    print("N-gram Statistics:")
    for n in range(100):
        try:
            term_count = len(voca[n])
            print(f"n={n}: {term_count} terms")
            for t in voca[n]:
                print(t)
                break
        except KeyError as e:
            pass


if __name__ == "__main__":
    main()
    show_cnt()
