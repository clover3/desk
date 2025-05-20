import os
import pickle

import numpy as np

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.path_helper import get_rp_path, get_split_subreddit_list


def load_mat_terms_pickled():
    pkl_path = get_rp_path("score_mat_and_infos.pkl")
    return pickle.load(open(pkl_path, "rb"))


def load_mat_terms(n_list) -> tuple[np.array, list[tuple], list[str]]:
    print("Loading data...")
    dir_name = "run6_10k_score"
    sb_len = None
    score_mat_list = []
    term_list_ex = []
    for n in n_list:
        score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
        score_mat_list.append(score_mat)
        if sb_len is None:
            sb_len = len(valid_sb_list)
        else:
            assert sb_len == len(valid_sb_list)
        term_list = load_run6_10k_terms(n)
        term_list_ex.extend(term_list)
    score_mat = np.concatenate(score_mat_list, axis=0)
    term_list = term_list_ex
    return score_mat, term_list, valid_sb_list
    # X = score_mat
    # X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    # correlations = X_norm @ X_norm
    # return correlations, term_list


def load_run_score_matrix(dir_name, n):
    subreddit_list = get_split_subreddit_list("train")
    score_list = []
    valid_sb_list = []
    for sb in subreddit_list:
        score_path = get_rp_path(dir_name, f"{sb}.{n}.pkl")
        if os.path.exists(score_path):
            scores = pickle.load(open(score_path, "rb"))
            score_list.append(scores)
            valid_sb_list.append(sb)
    score_mat = np.stack(score_list, axis=1)
    return score_mat, valid_sb_list
