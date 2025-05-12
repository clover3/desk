import os
import pickle

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.run6.score_analysis.show_pca import load_voca_list_str
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path



def save_mean_delta(voca: list[str], score_dir_name, metric, n):
    score_path = get_rp_path(score_dir_name, f"{metric}.{n}.pkl")
    med_score = pickle.load(open(score_path, "rb"))
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        score_path = get_rp_path(score_dir_name, f"{sb}.{n}.pkl")
        if os.path.exists(score_path):
            orig_scores = pickle.load(open(score_path, "rb"))
            scores = orig_scores - med_score

            l = list(zip(voca, scores))
            l.sort(key=lambda x: x[1], reverse=True)
            assert len(scores) == len(voca)
            text_out_path = get_rp_path(f"{score_dir_name}_{metric}_delta_score_text",
                                        f"{sb}.{n}.txt")
            make_parent_exists(text_out_path)

            with open(text_out_path, "w") as f:
                for term, s in l:
                    f.write("{}\t{}\n".format(term, s))


def main(n=1, metric="mean"):
    score_dir_name = "run6_voca"
    voca = load_voca_list_str(n)
    save_mean_delta(voca, score_dir_name, metric, n)



if __name__ == "__main__":
    fire.Fire(main)
