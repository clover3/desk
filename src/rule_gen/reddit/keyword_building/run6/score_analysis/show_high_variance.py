import numpy as np

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix


def main():
    n_list = list(range(1, 4))
    print("Loading data...")
    dir_name = "run6_10k_score"
    k = 0
    for n in n_list:
        score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
        churning_idx = valid_sb_list.index("churning")
        score_mat = np.concatenate([score_mat[:, :churning_idx], score_mat[:, churning_idx + 1:]], axis=1)
        valid_sb_list = valid_sb_list[:churning_idx] + valid_sb_list[churning_idx + 1:]
        print("score_mat.shape", score_mat.shape)
        print("N=", n)
        term_list = load_run6_10k_text(n)
        term_list = term_list
        score_mat = score_mat
        mean = np.mean(score_mat, axis=1)

        mod_indices,  = np.nonzero(mean > 0.7)
        var = np.var(score_mat, axis=1)
        print("var", var.shape)
        rank = np.argsort(var, axis=0)
        display_indices = [r for r in rank if r in mod_indices]
        cnt = 0
        ignore_pattern = ["http", "www.", "-", ".com", ".gov", ".org", ".co.", ">", "<"]
        # for idx in rank[:20]:
        for idx in display_indices:
            skip = False
            for t in ignore_pattern:
                if t in term_list[idx]:
                    skip = True
            if skip:
                continue
            cnt += 1
            if cnt > 10:
                break
            print("{} {} {}".format(term_list[idx], mean[idx], var[idx]))
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
