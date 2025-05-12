import json
import pickle

import tqdm

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_train_data_path_ex, get_rp_path


def map_voca_comment(
        comment_d: dict[str, tuple[str, int]],
        inv_index, voca: str):
    postings = inv_index[voca]
    if not postings:
        raise KeyError("No postings found for {}".format(voca))

    # If multiple occurrence?


def load_docs(split="train"):
    subreddit_list = get_split_subreddit_list(split)
    data_name = "train_data2"
    n_item = 2000

    comment_d: dict[str, tuple[str, int]] = {}
    for sb in tqdm.tqdm(subreddit_list):
        data_path = get_reddit_train_data_path_ex(data_name, sb, "train")
        data = read_csv(data_path)[:n_item]
        for idx, (text, label) in enumerate(data):
            data_id = "{}_{}_{}".format(data_name, sb, idx)
            comment_d[data_id] = text, int(label)

    return comment_d


def main():
    text_d_path = get_rp_path("run6_doc_map.pkl")
    text_d = pickle.load(open(text_d_path, "rb"))
    inv_index_path = get_rp_path("run6_inv_index2_train.pkl")
    inv_index_d = pickle.load(open(inv_index_path, "rb"))

    for n in range(1, 30):
        print("{} gram".format(n))
        topk_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
        voca: list = pickle.load(open(topk_path, "rb"))

        mapping_save_path = get_rp_path("run6_voca_doc_map", f"{n}.jsonl")
        make_parent_exists(mapping_save_path)
        f = open(mapping_save_path, "w")
        inv_index = inv_index_d[n]
        for (term_key, term_text, _) in tqdm.tqdm(voca):
            doc_names = inv_index[term_key]
            if doc_names:
                doc_name = next(iter(doc_names))
                text = text_d[doc_name]
                row = {"term": term_text, "doc_name": doc_name, "text": text}
                f.write(json.dumps(row) + "\n")
            else:
                print("No text found for {}".format(term_key))


if __name__ == "__main__":
    main()
