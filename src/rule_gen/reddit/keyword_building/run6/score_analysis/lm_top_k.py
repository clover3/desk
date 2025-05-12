import os
import pickle

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.path_helper import get_rp_path


def save_text():
    for n in range(1, 11):
        lm_prob_path = get_rp_path("run6_voca_lm_prob", f"{n}.pkl")
        lm_probs: list[tuple[tuple, str, float]] = pickle.load(open(lm_prob_path, "rb"))
        lm_probs.sort(key=lambda x: x[2], reverse=True)
        lm_prob_txt_path = get_rp_path("run6_voca_lm_prob", f"{n}.txt")

        with open(lm_prob_txt_path, "w") as f:
            for term, text, score in lm_probs:
                f.write(f"{text}\t{score}\n")


def save_pickl():
    for n in range(1, 11):
        k = 10000
        topk_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
        if os.path.exists(topk_path):
            continue
        lm_prob_path = get_rp_path("run6_voca_lm_prob", f"{n}.pkl")
        lm_probs: list[tuple[tuple, str, float]] = pickle.load(open(lm_prob_path, "rb"))
        lm_probs.sort(key=lambda x: x[2], reverse=True)
        lm_probs = lm_probs[:k]
        make_parent_exists(topk_path)
        pickle.dump(lm_probs, open(topk_path, "wb"))



if __name__ == "__main__":
    save_pickl()
