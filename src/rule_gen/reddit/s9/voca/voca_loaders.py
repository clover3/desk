import os
import pickle

import numpy as np

from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list


def load_voca_matrix():
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
    sel_indices = []
    new_voca = []
    for i, v in enumerate(T):
        if v > 2:
            sel_indices.append(i)
            new_voca.append(voca[i])
    score_mat_new = score_mat[sel_indices, :]
    return new_voca, score_mat_new, valid_sb_list


def load_voca_matrix_both(min_sb_freq=5, make_int=True):
    voca_d = {}
    for i in [1, 2]:
        voca_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                 "s9_ngram_93_voca", f"{i}.txt")
        print("Reading voca")
        voca = [l.strip() for l in open(voca_path, "r")]
        voca_d[i] = voca

    print("Reading scores")
    voca = voca_d[1] + voca_d[2]
    subreddit_list = get_split_subreddit_list("train")
    score_list = []
    valid_sb_list = []
    for sb in subreddit_list:
        score_pair = []
        for i in [1, 2]:
            score_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca_score",
                                      f"{sb}.{i}.pkl")
            if os.path.exists(score_path):
                scores = pickle.load(open(score_path, "rb"))
                if make_int:
                    scores_b = (np.array(scores) > 0.8).astype(int)
                else:
                    scores_b = np.array(scores)
                score_pair.append(scores_b)

        if len(score_pair) == 2:
            score = np.concatenate([score_pair[0], score_pair[1]], axis=0)
            score_list.append(score)
            valid_sb_list.append(sb)

    score_mat = np.stack(score_list, axis=1)
    T = np.sum(score_mat, axis=1)
    sel_indices = []
    new_voca = []
    for i, v in enumerate(T):
        if v >= min_sb_freq:
            sel_indices.append(i)
            new_voca.append(voca[i])
    score_mat_new = score_mat[sel_indices, :]
    return new_voca, score_mat_new, valid_sb_list
