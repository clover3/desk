import fire
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix
from rule_gen.reddit.keyword_building.run6.score_analysis.show_pca import load_voca_list_str
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def main(n_comp=2):
    n_list = [1, 2]
    print("Loading data...")
    dir_name = "run6_voca"
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
        term_list = load_voca_list_str(n)
        term_list_ex.extend(term_list)

    term_list = term_list_ex
    print("term_list_ex", len(term_list_ex))
    score_mat = np.concatenate(score_mat_list, axis=0)
    print("score_mat", score_mat.shape)
    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    delta_mat = score_mat - mean_mat
    print(f"Data shape: {score_mat.shape}")

    model = PCA(n_components=n_comp)
    W = model.fit_transform(delta_mat)
    H = model.components_
    mat_estimate = np.dot(W, H)
    assert len(term_list) == W.shape[0]
    rmse = np.sqrt(mean_squared_error(delta_mat, mat_estimate))
    print("RMSE={}".format(rmse))
    rmse = np.sqrt(mean_squared_error(delta_mat, np.zeros_like(delta_mat)))
    print("Baseline RMSE={}".format(rmse))
    run_show_pca(W, H, term_list, valid_sb_list)


if __name__ == '__main__':
    fire.Fire(main)
