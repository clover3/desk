import pickle

import fire
import numpy as np

from rule_gen.reddit.keyword_building.run6.score_analysis.compute_avg import load_run6_score_matrix
from rule_gen.reddit.path_helper import get_rp_path


def top_one_diff(mat):
    rank = np.argsort(mat, axis=1)[:, ::-1]

    out_arr = []
    for i in range(mat.shape[0]):
        v = mat[i, rank[i, 0]] - mat[i, rank[i, 1]]
        out_arr.append(v)

    return np.array(out_arr), rank[:, 0]

def get_unique_skip_indices(score_mat):
    diff_arr, _ = top_one_diff(score_mat)
    t = 0.1
    


def main(n=1):
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca = voca_d[n]
    score_mat, valid_sb_list = load_run6_score_matrix(n)
    print("Computing top one_diff")
    diff_arr, top_indice = top_one_diff(score_mat)
    t = 0.1
    m = np.count_nonzero(diff_arr > t)
    print(f"{m} terms have gap over {t}")
    # rank = np.argsort(diff_arr)[::-1]
    # for j in range(200):
    #     i = rank[j]
    #     top_i = top_indice[i]
    #     print(voca[i], diff_arr[i], valid_sb_list[top_i])
    #
    #


if __name__ == "__main__":
    fire.Fire(main)
