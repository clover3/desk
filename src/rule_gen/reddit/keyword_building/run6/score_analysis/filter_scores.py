import os
import pickle

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms_column
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main(n=1):
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca = voca_d[n]

    top_terms = load_run6_10k_terms_column(0, n)

    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        text_out_path = get_rp_path("run6_10k_score",
                                    f"{sb}.{n}.txt")
        pkl_path = get_rp_path("run6_10k_score",
                                    f"{sb}.{n}.pkl")
        score_path = get_rp_path("run6_voca", f"{sb}.{n}.pkl")
        if os.path.exists(score_path):
            scores = pickle.load(open(score_path, "rb"))
            score_d = dict(zip(voca, scores))
            sel_scores = []
            sel_voca_scores = []
            for t in top_terms:
                sel_scores.append(score_d[t])
                sel_voca_scores.append((t, score_d[t]))

            pickle.dump(sel_scores, open(pkl_path, "wb"))
            sel_voca_scores.sort(key=lambda x: x[1], reverse=True)
            assert len(scores) == len(voca)
            make_parent_exists(text_out_path)
            with open(text_out_path, "w") as f:
                for t, s in sel_voca_scores:
                    if n == 1:
                        term = t
                    else:
                        term = " ".join(t)
                    f.write("{}\t{}\n".format(term, s))


if __name__ == "__main__":
    fire.Fire(main)
