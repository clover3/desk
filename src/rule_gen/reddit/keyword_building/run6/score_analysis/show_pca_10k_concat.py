import fire
import numpy as np
from sklearn.decomposition import PCA, SparsePCA, NMF
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def main(n_comp=2):
    n_list = list(range(1, 4))
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
        term_list = load_run6_10k_text(n)
        term_list_ex.extend(term_list)

    term_list = term_list_ex
    print("term_list_ex", len(term_list_ex))
    score_mat = np.concatenate(score_mat_list, axis=0)
    print("score_mat", score_mat.shape)
    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    use_delta = True
    if use_delta:
        delta_mat = score_mat - mean_mat
        target_mat = delta_mat.transpose()
    else:
        target_mat = score_mat.transpose()
    print(f"Data shape: {target_mat.shape}")
    # model = SparsePCA(n_components=n_comp)
    model = PCA(n_components=n_comp)
    # U *= sqrt(X.shape[0] - 1)
    W = model.fit_transform(target_mat)
    H = model.components_

    norm_w = True
    if norm_w:
        column_norms = np.linalg.norm(W, axis=0, keepdims=True)
        W = W / column_norms
        H = column_norms.T * H

    mat_estimate = np.dot(W, H)
    rmse = np.sqrt(mean_squared_error(target_mat, mat_estimate))
    print("RMSE={}".format(rmse))
    target_mean = np.mean(target_mat, axis=1, keepdims=True)
    baseline = np.tile(target_mean, [1, target_mat.shape[1]])
    rmse = np.sqrt(mean_squared_error(target_mat, baseline))
    print("Baseline RMSE={}".format(rmse))
    run_show_pca(W, H, valid_sb_list, term_list)


if __name__ == '__main__':
    fire.Fire(main)
