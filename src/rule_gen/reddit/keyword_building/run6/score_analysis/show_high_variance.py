import numpy as np

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix

def main():
    n_list = list(range(1, 10))
    print("Loading data...")
    dir_name = "run6_10k_score"
    k = 0
    top_k = 200
    for n in n_list:
        score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
        print("N=", n)
        term_list = load_run6_10k_text(n)
        term_list = term_list[:top_k]
        # row_sorted = np.sort(score_mat, axis=1)
        # print("row_sorted", row_sorted.shape)
        # trimmed = row_sorted[:, k:-k]
        score_mat = score_mat[:top_k]
        var = np.var(score_mat, axis=1)
        print("var", var.shape)
        rank = np.argsort(var, axis=0)[::-1]
        cnt = 0
        ignore_pattern = ["http", "www.", "-", ".com", ".gov", ".org", ".co."]
        for idx in rank:
            skip = False
            for t in ignore_pattern:
                if t in term_list[idx]:
                    skip = True
            if skip:
                continue
            cnt += 1
            if cnt > 10:
                break
            print(term_list[idx], var[idx])
            X = score_mat
            s = " ".join([valid_sb_list[j] for j in
                          range(len(valid_sb_list)) if X[idx, j] > 0.7])
            print("Moderated at ", s)
            s = " ".join([valid_sb_list[j] for j in
                          range(len(valid_sb_list)) if X[idx, j] < 0.3])
            print("Not moderated at ", s)
            print("----")

if __name__ == "__main__":
    main()
