import json
import os
import pickle
from collections import defaultdict
from typing import Set, Any

import tqdm
from nltk import ngrams
from transformers import BertTokenizer

from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    voca = defaultdict(set)
    base_model = 'bert-base-uncased'
    inv_index_path = get_rp_path("run6_inv_index2_train.pkl")
    text_d_path = get_rp_path("run6_doc_map.pkl")

    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    inv_index_map_d: dict[int, dict[Any, Set]] = {n: defaultdict(set) for n in range(1,31)}
    text_map = {}
    for sb in tqdm.tqdm(subreddit_list):
        print(sb)
        save_path = get_rp_path( "s9_ngram_93", f"{sb}.json")
        if not os.path.exists(save_path):
            continue

        items = json.load(open(save_path))
        for i, res in enumerate(items):
            data_name = f"{sb}_{i}"
            text_map[data_name] = res["text"]
            for k in res["strong_sub_texts"]:
                sub_text: str = k["sub_text"]
                tokens = tokenize_fn(sub_text)
                inv_index_map = inv_index_map_d[1]
                for t in tokens:
                    inv_index_map[t].add(data_name)
                    voca[1].add(t)

                n = 2
                while n <= len(tokens) and n <= 30:
                    inv_index_map = inv_index_map_d[n]
                    for seq in ngrams(tokens, n):
                        voca[n].add(tuple(seq))
                        inv_index_map[tuple(seq)].add(data_name)
                    n += 1

    pickle.dump(inv_index_map_d, open(inv_index_path, "wb"))
    pickle.dump(text_map, open(text_d_path, "wb"))


if __name__ == "__main__":
    main()
