import numpy as np
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms_pickled
from rule_gen.cpath import output_root_path
import os
from scipy.stats import pearsonr

def main():
    rate_file = os.path.join(output_root_path, "reddit", "all_mod.txt")
    pos_rate_list = []
    for line in open(rate_file, "r"):
        sb, pos, neg = line.split("\t")
        pos = int(pos)
        neg = int(neg)
        pos_rate = pos / (pos + neg)
        pos_rate_list.append(pos_rate)

    print(len(pos_rate_list))

    score_mat, term_list, valid_sb_list = load_mat_terms_pickled()
    term = "mod"
    idx = term_list.index(term)
    print("For term", term)

    churning_idx = valid_sb_list.index("churning")
    scores = score_mat[idx]
    print("Mod score min/max", np.min(scores), np.max(scores), np.mean(scores), np.std(scores))
    print(len(scores))
    print(pearsonr(scores, pos_rate_list))

    #
    #
    # for t in [0.9, 0.8, 0.7, 0.6, 0]:
    #     indices,= np.nonzero(scores > t)
    #     print(t, len(indices), [valid_sb_list[i] for i in indices])


if __name__ == "__main__":
    main()