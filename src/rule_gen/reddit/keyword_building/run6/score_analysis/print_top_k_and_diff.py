import numpy as np

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms_pickled, \
    load_run_score_matrix


def main():
    dir_name = "run6_10k_score"
    n = 3
    score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
    term_list = load_run6_10k_terms(n)

    sb_index = valid_sb_list.index("fantasyfootball")
    mean_scores = np.mean(score_mat, axis=1)
    delta = score_mat[:, sb_index] - mean_scores
    ranks = np.argsort(delta)[::-1]

    for i in ranks[:10]:
        term_text = " ".join(term_list[i])
        l = "{}\t{:.2f}\t{:.2f}".format(
            term_text, score_mat[i, sb_index], mean_scores[i])

        print(l)


if __name__ == "__main__":
    main()
