from collections import Counter

import tqdm
import numpy as np
import pickle

from rule_gen.reddit.keyword_building.run6.gpt_norms.response_norm_match import load_manual_norm_voca
from rule_gen.reddit.path_helper import get_rp_path, get_split_subreddit_list


def run_for(norm_diff_save_dir, summary_path):
    subreddit_list = get_split_subreddit_list("train")
    voca = load_manual_norm_voca()
    print(voca)
    voca = list(voca)
    voca.sort()
    head = [""] + voca
    table = [head]
    for sb in tqdm.tqdm(subreddit_list):
        n_item = 0
        all_bow = Counter()
        for n in range(1, 10):
            norm_diff_bow_save_path = get_rp_path(norm_diff_save_dir, f"{sb}.{n}.pkl")
            norm_diff_bow = pickle.load(open(norm_diff_bow_save_path, "rb"))
            for k, v in norm_diff_bow.items():
                all_bow[k] += v
            n_item += 1
        bow = Counter({k: v / n_item for k, v in all_bow.items()})

        row = [sb]
        for k in voca:
            row.append("{0:.2f}".format(bow[k]))
        table.append(row)

    with open(summary_path, "w") as f:
        for row in table:
            f.write("\t".join(row) + "\n")



def main():
    summary_path = get_rp_path("run6_man_norm_diff_summary.tsv")
    norm_diff_save_dir = "run6_man_norm_diff"
    run_for(norm_diff_save_dir, summary_path)


def main2():
    summary_path = get_rp_path("run6_man_norm_summary.tsv")
    norm_diff_save_dir = "run6_man_norm"
    run_for(norm_diff_save_dir, summary_path)


if __name__ == "__main__":
    main()