import os
import pickle

import fire
import numpy as np

from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix
from rule_gen.reddit.path_helper import get_rp_path


def voca_join(sb, n=1):
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca: list = voca_d[n]
    score_path = get_rp_path("run6_voca", f"{sb}.{n}.pkl")
    if os.path.exists(score_path):
        scores = pickle.load(open(score_path, "rb"))
        l = list(zip(voca, scores))
        l.sort(key=lambda x: x[1], reverse=True)
        assert len(scores) == len(voca)
        text_out_path = get_rp_path("run6_voca_score_text",
                                    f"{sb}.{n}.txt")

        with open(text_out_path, "w") as f:
            for t, s in l:
                if n == 1:
                    term = t
                else:
                    term = " ".join(t)
                f.write("{}\t{}\n".format(term, s))


def load_run6_score_matrix(n):
    dir_name = "run6_voca"
    score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
    return score_mat, valid_sb_list


def main(n=1):
    voca_path = get_rp_path("run6_voca.pkl")
    score_mat, valid_sb_list = load_run6_score_matrix(n)

    score_mean = np.mean(score_mat, axis=1)
    mean_score_path = get_rp_path("run6_voca", f"mean.{n}.pkl")
    pickle.dump(score_mean, open(mean_score_path, "wb"))
    voca_join("mean", n)

    score_median = np.median(score_mat, axis=1)
    median_score_path = get_rp_path("run6_voca", f"median.{n}.pkl")
    pickle.dump(score_median, open(median_score_path, "wb"))
    voca_join("median", n)



if __name__ == "__main__":
    fire.Fire(main)
