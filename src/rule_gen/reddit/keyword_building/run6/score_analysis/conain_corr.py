import fire
import numpy as np
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix


def main():
    n_list = list(range(1, 10))
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
    use_delta = False
    if use_delta:
        delta_mat = score_mat - mean_mat
        target_mat = delta_mat
    else:
        target_mat = score_mat

    X = target_mat
    X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)

    while True:
        try:
            cands = []
            term = input("Enter a term: ")
            src_idx = term_list.index(term)

            for idx, t in enumerate(term_list):
                if term in t:
                    cands.append(idx)
            correlations = X_norm @ X_norm[src_idx]
            correlations = correlations[cands]
            indices = np.argsort(correlations)[::-1]

            print("top k")
            for i in indices[:50]:
                idx = cands[i]
                s = " ".join(["{0:.1f}".format(v) for v in score_mat[idx, :]])
                print(term_list[idx], correlations[i], s)

            print("bottom k")
            for i in indices[::-1][:50]:
                idx = cands[i]
                s = " ".join(["{0:.1f}".format(v) for v in score_mat[idx, :]])
                print(term_list[idx], correlations[i], s)

            print("-------------")
        except (IndexError, ValueError) as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(main)
