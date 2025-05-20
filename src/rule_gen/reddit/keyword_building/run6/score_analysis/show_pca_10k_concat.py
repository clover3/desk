import fire
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def is_url_or_dash(text):
    ignore_pattern = ["http", "www.", "-", ".com", ".gov", ".org", ".co."]
    for t in ignore_pattern:
        if t in text:
            return True
    return False


def main(n_comp=2):
    n_list = list(range(2, 5))
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

    term_list_new = []
    mat_sel_indices = []
    for i, term in enumerate(term_list_ex):
        if not is_url_or_dash(term):
            term_list_new.append(term)
            mat_sel_indices.append(i)

    term_list = term_list_new
    score_mat = score_mat[mat_sel_indices]

    mean_mat = np.mean(score_mat, axis=1, keepdims=True)
    use_delta = True
    if use_delta:
        delta_mat = score_mat - mean_mat
        target_mat = delta_mat.transpose()
    else:
        target_mat = score_mat.transpose()
    print(f"Data shape: {target_mat.shape}")
    # model = SparsePCA(n_components=n_comp, max_iter=2)
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
    if n_comp == 1:
        n_show_features = 400
    else:
        n_show_features = 40
    run_show_pca(W, H, valid_sb_list, term_list, n_show_features=n_show_features)


if __name__ == '__main__':
    fire.Fire(main)
