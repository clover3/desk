import pickle
from collections import Counter

from rule_gen.reddit.path_helper import get_rp_path


def load_tf_df():
    tf_pkl_path = get_rp_path("all_tf_df.pkl")
    all_tf_df = pickle.load(open(tf_pkl_path, "rb"))
    acc_df = Counter()
    acc_tf = Counter()
    for sb, tf, df in all_tf_df:
        for k, v in df.items():
            acc_df[k] += v

        for k, v in df.items():
            acc_tf[k] += v
    return acc_df, acc_tf


def main():
    acc_df, acc_tf = load_tf_df()
    n = 1
    score_path = get_rp_path("run6_voca_lm_prob", f"{n}.pkl")
    voca_prob = pickle.load(open(score_path, "rb"))
    voca_prob.sort(key=lambda x: x[2], reverse=True)

    from_lm = [term for term, _, lm_prob in voca_prob[:10000]]

    ranked = []
    for term, _, _ in voca_prob:
        ranked.append((term, acc_df[term]))
    ranked.sort(key=lambda x: x[1], reverse=True)
    from_tf = [term for term, _ in ranked[:10000]]

    t_new = [t for t in from_tf if t not in from_lm]
    n_common = len(set(from_lm).intersection(from_tf))
    print("{} of 10000 are common".format(n_common))
    print("New terms", t_new)


if __name__ == "__main__":
    main()
