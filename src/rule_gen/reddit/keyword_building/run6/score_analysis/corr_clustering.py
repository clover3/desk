import pickle

from rule_gen.reddit.path_helper import get_rp_path


def load_top_terms():
    dir_name = "run6_10k_score"
    sb = "mean"
    n = 1
    score_path = get_rp_path(dir_name, f"{sb}.{n}.pkl")
    scores = pickle.load(open(score_path, "rb"))
    voca_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
    voca = pickle.load(open(voca_path, "rb"))
    return voca, scores



def main():
    # model = LabelSpreading()
    # model.fit(graph)
    pass

if __name__ == "__main__":
    main()