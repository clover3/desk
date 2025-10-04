import pickle

from rule_gen.reddit.path_helper import get_rp_path


def load_top_terms():
    dir_name = "sb_term_scores"
    sb = "mean"
    n = 1
    score_path = get_rp_path(dir_name, f"{sb}.{n}.pkl")
    scores = pickle.load(open(score_path, "rb"))
    voca_path = get_rp_path("top_10k_voca", f"{n}.pkl")
    voca = pickle.load(open(voca_path, "rb"))
    return voca, scores



def main():
    # model = LabelSpreading()
    # model.fit(graph)
    pass

if __name__ == "__main__":
    main()