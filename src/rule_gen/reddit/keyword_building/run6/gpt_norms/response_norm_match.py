import tqdm
import pickle
from collections import Counter, defaultdict

from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.term_norm_match import load_doc_id_to_response

import re

from rule_gen.reddit.path_helper import get_rp_path



def load_norm_voca():
    path = get_rp_path(f"norm_voca", "asterisk_terms.pkl")
    voca1 = pickle.load(open(path, "rb"))

    path = get_rp_path(f"norm_voca", "unigram.pkl")
    voca2 = pickle.load(open(path, "rb"))
    return voca1 + voca2


def main1():
    doc_id_to_res = load_doc_id_to_response()
    voca = load_norm_voca()
    v_low = {v.lower() for v in voca}
    output = []
    for doc_id, text in tqdm.tqdm(doc_id_to_res.items()):
        bow = Counter()
        t_low = text.lower()
        for v in v_low:
            if v.lower() in t_low:
                bow[v] += 1

        output.append((doc_id, bow))

    path = get_rp_path(f"norm_voca", "doc_id_bow_norm.pkl")
    pickle.dump(output, open(path, "wb"))


def main():
    doc_id_to_res = load_doc_id_to_response()
    v_low = load_manual_norm_voca()
    print(v_low)
    output = []
    for doc_id, text in tqdm.tqdm(doc_id_to_res.items()):
        bow = Counter()
        t_low = text.lower()
        for v in v_low:
            pattern = v.lower()
            if len(v) <= 4:
                pattern = " {} ".format(v)

            if pattern in t_low:
                bow[v] += 1

        output.append((doc_id, bow))

    path = get_rp_path(f"norm_voca", "doc_id_bow_man_norm.pkl")
    pickle.dump(output, open(path, "wb"))


def load_manual_norm_voca():
    path = get_rp_path(f"norm_voca", "man.txt")
    note = {}
    voca = []
    with open(path, "r") as f:
        for line in f:
            tokens = line.split("\t")
            term = tokens[0]
            if len(tokens) > 1:
                note[term] = tokens[1:]
            voca.append(term.strip())
    v_low = {v.lower() for v in voca}
    return v_low


if __name__ == "__main__":
    main()