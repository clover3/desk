import json
import os
import pickle
from collections import defaultdict

from nltk import ngrams
from transformers import BertTokenizer

from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    voca = defaultdict(set)
    base_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    rev_map = defaultdict(dict)

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
                        key = tuple(seq)
                        voca[n].add(key)
                        rev_map[n][key] = sub_text
                    n += 1
    save_path = get_rp_path("run6_voca_rev_src.pkl")
    pickle.dump(rev_map, open(save_path, "wb"))


if __name__ == "__main__":
    main()
