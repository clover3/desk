import os
import pickle

import fire

from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main(n=1):
    voca_path = get_rp_path("run6_voca_l.pkl")
    voca_d = pickle.load(open(voca_path, "rb"))
    voca = voca_d[n]
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
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


if __name__ == "__main__":
    fire.Fire(main)
