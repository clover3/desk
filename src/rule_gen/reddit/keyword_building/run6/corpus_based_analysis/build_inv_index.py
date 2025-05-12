import pickle
from collections import defaultdict
from typing import Iterable

import tqdm
from transformers import BertTokenizer

from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex, get_rp_path


def build_inv_index(itr: Iterable[tuple[str, list[str]]]):
    inv_index = defaultdict(list)
    for doc_id, tokens in itr:
        for token in set(tokens):
            inv_index[token].append(doc_id)
    return inv_index


def main():
    split = "train"
    base_model = 'bert-base-uncased'
    subreddit_list = get_split_subreddit_list(split)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    data_name = "train_data2"
    n_item = 2000

    def data_iter():
        for sb in tqdm.tqdm(subreddit_list):
            data_path = get_reddit_train_data_path_ex(data_name, sb, "train")
            data = read_csv(data_path)[:n_item]
            for idx, (text, _label) in enumerate(data):
                data_id = "{}_{}_{}".format(data_name, sb, idx)
                tokens = tokenize_fn(text)
                yield data_id, tokens

    inv_index = build_inv_index(data_iter())
    inv_index_path = get_rp_path("run6_inv_index_train.pkl")
    inv_index_path = inv_index_path
    pickle.dump(inv_index, open(inv_index_path, "wb"))


if __name__ == "__main__":
    main()
