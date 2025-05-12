import os.path
import pickle

import numpy as np
import fire

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix
from rule_gen.reddit.keyword_building.run6.score_analysis.median_delta import save_mean_delta
from rule_gen.reddit.path_helper import get_rp_path


def main(n=1, metric="mean"):
    score_dir_name = "run6_10k_score"

    mean_score_path = get_rp_path(score_dir_name, f"{metric}.{n}.pkl")
    if not os.path.exists(mean_score_path):
        score_mat, valid_sb_list = load_run_score_matrix(score_dir_name, n)
        if metric == "mean":
            score_mean = np.mean(score_mat, axis=1)
            pickle.dump(score_mean, open(mean_score_path, "wb"))

        elif metric == "median":
            score_median = np.median(score_mat, axis=1)
            pickle.dump(score_median, open(mean_score_path, "wb"))

    voca = load_run6_10k_text(n)
    save_mean_delta(voca, score_dir_name, metric, n)


if __name__ == "__main__":
    fire.Fire(main)
