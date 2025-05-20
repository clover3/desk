import json

import numpy as np
import sklearn
import sklearn.metrics

from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms_pickled
from rule_gen.reddit.path_helper import get_rp_path


def main():
    # Initialize voca -> cluster mapping
    # Load all voca.
    score_mat, term_list, valid_sb_list = load_mat_terms_pickled()

    score_path = get_rp_path("clustering", f"100.json")
    j = json.load(open(score_path, "r"))
    mat_list = []
    labels = []
    for e in j:
        no = e["cluster_no"]
        term_indices = []
        for t in e["terms"]:
            if type(t) == str:
                idx = term_list.index(t)
            else:
                t = tuple(t)
                idx = term_list.index(t)
            term_indices.append(idx)

        score_sub_mat = score_mat[term_indices]
        print("score_sub_mat", score_sub_mat.shape)
        mat_list.append(score_sub_mat)
        labels.extend([no] * len(term_indices))

    X = np.concatenate(mat_list, axis=0)
    s = sklearn.metrics.silhouette_score(X, labels, metric='euclidean')
    print("silhouette_score", s)


if __name__ == '__main__':
    main()
    # if e["average"] < 0.5:
    #     skip_n += 1
    #     continue
