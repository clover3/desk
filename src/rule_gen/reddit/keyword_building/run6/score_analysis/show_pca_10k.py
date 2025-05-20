import fire
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def main(n_comp=2, n=1):
    print("Loading data...")
    dir_name = "run6_10k_score"
    score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)

    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    delta_mat = score_mat - mean_mat
    term_list = load_run6_10k_text(n)

    delta_mat = delta_mat.transpose()
    print(f"Data shape: {delta_mat.shape}")
    model = PCA(n_components=n_comp)
    W = model.fit_transform(delta_mat)
    H = model.components_
    mat_estimate = np.dot(W, H)
    assert len(term_list) == W.shape[0]
    rmse = np.sqrt(mean_squared_error(delta_mat, mat_estimate))
    print("RMSE={}".format(rmse))
    rmse = np.sqrt(mean_squared_error(delta_mat, np.zeros_like(delta_mat)))
    print("Baseline RMSE={}".format(rmse))
    run_show_pca(W, H, valid_sb_list, term_list)


if __name__ == '__main__':
    fire.Fire(main)
