import fire
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.score_analysis.compute_avg import load_run6_score_matrix
from rule_gen.reddit.keyword_building.run6.score_analysis.score_distrib import top_one_diff
from rule_gen.reddit.keyword_building.run6.score_analysis.show_pca import load_voca_list_str
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def main(n_comp = 2, n = 1):
    print("Loading data...")

    score_mat, valid_sb_list = load_run6_score_matrix(n)
    diff_arr, _ = top_one_diff(score_mat)
    print(diff_arr.shape)
    select = diff_arr < 0.1
    select_i = np.nonzero(select)
    score_mat = score_mat[select_i]

    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    delta_mat = score_mat - mean_mat
    row_names = load_voca_list_str(n)
    new_row_names = []
    select_i = select_i[0]
    for i in select_i:
        new_row_names.append(row_names[i])
    print("{} terms selected from {}".format(len(new_row_names), len(row_names)))
    row_names = new_row_names
    print(f"Data shape: {score_mat.shape}")


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
