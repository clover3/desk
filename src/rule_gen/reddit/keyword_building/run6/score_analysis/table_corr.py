import fire
import numpy as np
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix


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
    print(f"Data shape: {target_mat.shape}")
    print(" ".join(valid_sb_list))
    X = target_mat
    while True:
        try:
            term = input("Enter term or index: ")
            try:
                idx = int(term)
                term = term_list[idx]
            except ValueError:
                idx = term_list.index(term)
            X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
            correlations = X_norm @ X_norm[idx]
            indices = np.argsort(correlations)[::-1]
            print("Target term {} ({})".format(term, idx))
            conditions = [
                (lambda x: x >= 0.7,"Moderated (>0.7)" ),
                (lambda x: 0.7 > x >= 0.5,"Moderated (>0.5)" ),
                (lambda x: 0.5 > x >= 0.3,"Not moderated (>0.3)" ),
                (lambda x: 0.3 > x >= 0,"Not moderated (<0.3)" )
            ]
            for fn, desc in conditions:
                s = " ".join([valid_sb_list[j] for j in
                              range(len(valid_sb_list)) if fn(X[idx, j])])
                print(desc, s)
            # s = " ".join([valid_sb_list[j] for j in
            #               range(len(valid_sb_list)) if X[idx, j] > 0.7])
            # print("Moderated (>0.7) at ", s)
            # s = " ".join([valid_sb_list[j] for j in
            #               range(len(valid_sb_list)) if X[idx, j] < 0.3])
            # print("Not moderated at ", s)

            for i in range(100):
                idx = indices[i]
                s = " ".join(["{0:.1f}".format(v) for v in score_mat[idx, :]])
                print(term_list[idx], correlations[idx], s)
            print("-------------")
        except (IndexError, ValueError) as e:
            print(e)


if __name__ == '__main__':
    fire.Fire(main)
