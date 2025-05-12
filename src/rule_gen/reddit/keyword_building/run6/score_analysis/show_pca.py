import pickle
from typing import Optional

import fire
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.score_analysis.compute_avg import load_run6_score_matrix
from rule_gen.reddit.path_helper import get_rp_path
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def load_voca_list(n) -> list[str | tuple]:
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca = voca_d[n]
    return voca


def load_voca_list_str(n) -> list[str]:
    voca = load_voca_list(n)
    output: list[str] = []
    for t in voca:
        if n == 1:
            term = t
        else:
            term = " ".join(t)
        output.append(term)
    return output



def main(n_comp = 2):
    print("Loading data...")
    n = 1
    score_mat, valid_sb_list = load_run6_score_matrix(n)
    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    delta_mat = score_mat - mean_mat
    print(f"Data shape: {score_mat.shape}")
    row_names = load_voca_list_str(n)
    print("{} terms".format(len(row_names)))


    model = PCA(n_components=n_comp)
    W = model.fit_transform(delta_mat)
    H = model.components_
    mat_estimate = np.dot(W, H)
    rmse = np.sqrt(mean_squared_error(delta_mat, mat_estimate))
    print("RMSE={}".format(rmse))
    rmse = np.sqrt(mean_squared_error(delta_mat, np.zeros_like(delta_mat)))
    print("Baseline RMSE={}".format(rmse))
    run_show_pca(W, H, row_names, valid_sb_list)


if __name__ == '__main__':
    fire.Fire(main)
