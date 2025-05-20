import pickle
from collections import Counter

from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms
from rule_gen.reddit.path_helper import get_rp_path



def main():
    n_list = list(range(1, 10))
    score_mat, term_list, valid_sb_list = load_mat_terms(n_list)
    pkl_path = get_rp_path("score_mat_and_infos.pkl")
    obj = (score_mat, term_list, valid_sb_list)
    pickle.dump(obj, open(pkl_path, "wb"))


if __name__ == "__main__":
    main()