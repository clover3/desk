import os
import pickle

import numpy as np

from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


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
