import pickle

from rule_gen.reddit.path_helper import get_rp_path


def load_top_10k_term(n):
    return load_top_10k_voca_column(0, n)


def load_top_10k_voca_column(col_i, n):
    topk_path = get_rp_path("top_10k_voca", f"{n}.pkl")
    voca = pickle.load(open(topk_path, "rb"))
    term_list = [e[col_i] for e in voca]
    return term_list


def load_top_10k_text(n):
    return load_top_10k_voca_column(1, n)
