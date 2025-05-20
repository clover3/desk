import numpy as np
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix
from rule_gen.reddit.path_helper import get_rp_path


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

    print("term_list_ex", len(term_list_ex))
    score_mat = np.concatenate(score_mat_list, axis=0)
    print("score_mat", score_mat.shape)
    score_path = get_rp_path("run6_sb_mean.txt")
    mean_mat = np.mean(score_mat, axis=0)
    paired = list(zip(valid_sb_list, mean_mat))
    paired.sort(key=lambda x: x[1], reverse=True)
    f = open(score_path, "w")
    for sb, v in paired:
        f.write(str(sb) + "\t" + str(v) + "\n")


if __name__ == "__main__":
    main()


