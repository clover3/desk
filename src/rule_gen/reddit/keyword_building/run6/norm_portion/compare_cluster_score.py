import numpy as np
import json
import pickle
from collections import Counter

from scipy.stats import ttest_rel

from chair.list_lib import left
from chair.misc_lib import Averager
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.keyword_building.run6.norm_portion.compute_cluster_portion import cluster_topic
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms_pickled
from rule_gen.reddit.path_helper import load_subreddit_list, get_rp_path
from chair.tab_print import print_table


def main():
    n_list = list(range(1, 10))
    print("Loading scores from pickle")
    score_mat, term_list, valid_sb_list = load_mat_terms_pickled()


    print("Loading cluster info")
    score_path = get_rp_path("clustering", f"100.json")
    j = json.load(open(score_path, "r"))
    term_to_cluster_id = {}
    c_no_to_t_list = {}
    for e in j:
        no = e["cluster_no"]
        l = []
        for t in e["terms"]:
            if type(t) == str:
                term_to_cluster_id[(t,)] = no
                term_idx = term_list.index(t)
            else:
                term_to_cluster_id[tuple(t)] = no
                term_idx = term_list.index(tuple(t))
            l.append(term_idx)
        c_no_to_t_list[no] = l

    print("Loading cluster topic")

    c_topics: dict[int, str] = cluster_topic()
    c_list = []
    for c_id, name in c_topics.items():
        if "personal attack" in name:
            c_list.append(c_id)


    # c1 = c_list[0]
    # c1_terms = c_no_to_t_list[c1]
    churning_idx = 3
    n_sb = len(valid_sb_list)

    norm_list = []
    for c_id in c_list:
        c1_terms = c_no_to_t_list[c_id]
        c_name = c_topics[c_id]
        mean = np.mean(score_mat[c1_terms])
        std = np.std(np.mean(score_mat[c1_terms], axis=0))
        norm_list.append((c_id, mean, std))

    norm_list.sort(key=lambda x: x[1], reverse=True)

    delta_d = {}

    for c_id, mean, std in norm_list:
        c1_terms = c_no_to_t_list[c_id]
        c_name = c_topics[c_id]
        print(f"{c_name} cluster={c_id} {mean:.2f} ({std:.2f})")
        per_sb_mean = []
        for sb_i in range(n_sb):
            if sb_i == churning_idx:
                continue
            scores1 = score_mat[c1_terms, sb_i]
            row = (valid_sb_list[sb_i], np.mean(scores1) - mean)
            delta_d[c_id, sb_i] = np.mean(scores1) - mean
            per_sb_mean.append(row)

        per_sb_mean.sort(key=lambda x: x[1], reverse=True)

        st = 0.9
        step = 0.1
        while st > 0:
            bot, top = st, st+step
            sb_list = [t[0] for t in per_sb_mean if bot <= t[1] < top ]
            if sb_list:
                s = " ".join(sb_list)
                print("{:.2f}~{:.2f}".format(bot, top), s)
            # print("{:.2f}~{:.2f}".format(cur[0][1], cur[-1][1]), s)
            st -= step

    for c_id, mean, std in norm_list:
        c1_terms = c_no_to_t_list[c_id]
        c_name = c_topics[c_id]
        print(f"{c_name} cluster={c_id} {mean:.2f} ({std:.2f})")

    table = []
    head = [""] + c_list
    table.append(head)
    for sb_i in range(n_sb):
        if sb_i == churning_idx:
            continue
        row = [valid_sb_list[sb_i]]
        for c_id, _, _ in norm_list:
            d = delta_d[c_id, sb_i]
            row.append(f"{d:.2f}")
        table.append(row)

    print_table(table)



    #     for sb_2 in range(sb_1+1, n_sb):
    #         scores2 = score_mat[c1_terms, sb_2]
    #         print(scores1)
    #         print(scores2)
    #         t, p = ttest_rel(scores1, scores2)
    #         print(sb_1, sb_2, t, p)
    #         break
    #     break
    # # scores1 = score_mat[c1]



if __name__ == "__main__":
    main()