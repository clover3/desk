import os
import pickle
from collections import Counter, defaultdict

import numpy as np

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    voca_d = {}
    i = 1
    voca_path = os.path.join(output_root_path, "reddit", "rule_processing",
                             "s9_ngram_93_voca", f"{i}.txt")
    print("Reading voca")
    voca = [l.strip() for l in open(voca_path, "r")]
    voca_d[i] = voca

    subreddit_list = get_split_subreddit_list("train")
    score_list = []
    valid_sb_list = []
    for sb in subreddit_list:
        score_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca_score",
                                  f"{sb}.{i}.pkl")
        if os.path.exists(score_path):
            valid_sb_list.append(sb)
            scores = pickle.load(open(score_path, "rb"))
            scores_b = (np.array(scores) > 0.8).astype(int)
            score_list.append(scores_b)

    score_mat = np.stack(score_list, axis=1)
    T = np.sum(score_mat, axis=1)
    counter = Counter()
    rev_indices = defaultdict(list)
    for i, v in enumerate(T):
        counter[v] += 1
        rev_indices[v].append(i)

    for i in range(score_mat.shape[1]):
        ex_list = []
        for j in rev_indices[i][:4]:
            ex_list.append(voca[j])
        print(i, counter[i], ex_list)

    per_sb = np.sum(score_mat, axis=0)
    entries = []
    for i, sb in enumerate(valid_sb_list):
        e = (sb, per_sb[i])
        entries.append(e)
    entries.sort(key=lambda e: e[1], reverse=True)
    for sb, n in entries:
        print(sb, n)


if __name__ == "__main__":
    main()
