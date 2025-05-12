import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

from rule_gen.reddit.keyword_building.run4.nmf_eval_common import run_nmf_eval
from rule_gen.reddit.s9.voca.voca_loaders import load_voca_matrix_both


def main():
    new_voca, X, valid_sb_list = load_voca_matrix_both()
    n_comp = 1
    print(f"Data shape: {X.shape}")
    X = X - np.mean(X, axis=1, keepdims=True)
    model = PCA(n_components=n_comp)

    baseline_msre = np.sqrt(mean_squared_error(X, np.zeros_like(X)))
    print("baseline_msre", baseline_msre)
    run_nmf_eval(model, X, n_comp)


if __name__ == "__main__":
    main()
