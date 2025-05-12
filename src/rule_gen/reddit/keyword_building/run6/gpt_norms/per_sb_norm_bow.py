import tqdm
import numpy as np



import pickle
from collections import Counter, defaultdict

from desk_util.io_helper import load_jsonl
from rule_gen.reddit.path_helper import get_rp_path, get_split_subreddit_list


def load_rp_pkl(dir_name, file_name):
    pkl_path = get_rp_path(dir_name, file_name)
    return pickle.load(open(pkl_path, "rb"))



def run_for(norm_diff_save_dir, voca_norm_map_dir, use_delta=True):
    score_dir_name = "run6_10k_score"
    subreddit_list = get_split_subreddit_list("train")
    for n in range(1, 10):
        print("{} gram".format(n))
        top_k_voca = load_rp_pkl("run6_voca_lm_prob_10k", f"{n}.pkl")
        voca_to_norm: dict[tuple, Counter] = dict(load_rp_pkl(voca_norm_map_dir, f"{n}.pkl"))
        score_list = []
        for sb in subreddit_list:
            scores: list[float] = load_rp_pkl(score_dir_name, f"{sb}.{n}.pkl")
            score_list.append(scores)
        score_mat = np.stack(score_list, axis=1)
        mean_score = np.mean(score_mat, axis=1)
        for sb in tqdm.tqdm(subreddit_list):
            norm_diff_bow = Counter()
            scores: list[float] = load_rp_pkl(score_dir_name, f"{sb}.{n}.pkl")
            if use_delta:
                target_scores = scores - np.array(mean_score)
            else:
                target_scores = scores

            for i, (term, _, _) in enumerate(top_k_voca):
                ds = target_scores[i]
                bow = voca_to_norm[term]
                d = sum(bow.values())
                for k, v in bow.items():
                    norm_diff_bow[k] += v/d * ds

            out_l = list(norm_diff_bow.items())
            out_l.sort(key=lambda x: x[0])
            norm_diff_bow_save_path = get_rp_path(norm_diff_save_dir, f"{sb}.{n}.pkl")
            pickle.dump(norm_diff_bow, open(norm_diff_bow_save_path, "wb"))
            norm_diff_bow_save_path = get_rp_path(norm_diff_save_dir, f"{sb}.{n}.txt")
            with open(norm_diff_bow_save_path, "w") as f:
                for k, v in out_l:
                    f.write(f"{k}\t{v:.3f}\n")


def main1():
    voca_norm_map_dir = "run6_voca_to_norm"
    norm_diff_save_dir = "run6_norm_diff"
    run_for(norm_diff_save_dir, voca_norm_map_dir)


def main():
    voca_norm_map_dir = "run6_voca_to_man_norm"
    norm_diff_save_dir = "run6_man_norm_diff"
    run_for(norm_diff_save_dir, voca_norm_map_dir)


def main3():
    voca_norm_map_dir = "run6_voca_to_man_norm"
    norm_diff_save_dir = "run6_man_norm"
    run_for(norm_diff_save_dir, voca_norm_map_dir, False)



if __name__ == "__main__":
    main()