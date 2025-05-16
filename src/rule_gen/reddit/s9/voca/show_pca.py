import ast
import random

import numpy as np
from sklearn.decomposition import PCA

from rule_gen.reddit.keyword_building.run4.nmf_eval_common import run_nmf_eval
from rule_gen.reddit.s9.voca.voca_loaders import load_voca_matrix_both


def run_show_pca(W, H, row_names, column_names,
                 n_show_columns=10, n_show_features=10):
    print("W", W.shape)
    print("H", H.shape)
    n_components = W.shape[1]
    head = [""] + list(map(str, range(n_components)))
    print("W sparsity", np.mean(W == 0))
    print("H sparsity", np.mean(H == 0))

    if H.shape[1] > 100:
        print("Component too large. print composition instead")
        print("\t".join(head))
        for row_i in range(W.shape[0]):
            s = "\t".join(["{0:.2f}".format(v) for v in W[row_i, :]])
            print(f"{row_names[row_i]}\t{s}")
    else:
        print("\t".join(head))
        for col_i in range(H.shape[1]):
            s = "\t".join(["{0:.2f}".format(v) for v in H[:, col_i]])
            print(f"{column_names[col_i]}\t{s}")

    for topic_idx, topic in enumerate(H):
        print(f"<--- Component {topic_idx} ----- >")
        print("over 0.1: ", np.count_nonzero(topic > 0.1))
        print("under -0.1: ", np.count_nonzero(topic < -0.1))

        print(f"\nComponent {topic_idx}: Top features (Terms)")
        top_features_idx = topic.argsort()[::-1][:n_show_features]
        print_features(column_names, top_features_idx, topic)

        component_scores = W[:, topic_idx]
        print(f"\nComponent {topic_idx} - Top Items (subreddit):")
        top_item_indices = component_scores.argsort()[::-1][:n_show_columns]  # Top 30 rows
        print_features(row_names, top_item_indices, component_scores)

        print(f"\nComponent {topic_idx}: Bottom features (Terms)")
        bottom_features_idx = topic.argsort()[:n_show_features]
        print_features(column_names, bottom_features_idx, topic)

        print(f"\nComponent {topic_idx} - Bottom Rows (subreddit):")
        bottom_item_indices = component_scores.argsort()[:n_show_columns]  # Bottom 30 rows
        bottom_item_indices = [i for i in bottom_item_indices if component_scores[i] < 0]
        print_features(row_names, bottom_item_indices, component_scores)
    #
    #
    # dim_len = H.shape[1]
    # dim_random = random.sample(range(dim_len), 30)
    # for i in dim_random:
    #     top_comps = H[:, i].argsort()[::-1][:n_show_features]
    #     s = "\t".join(["{0}: {1:.2f}".format(j, H[j, i]) for j in top_comps])
    #     print(i, column_names[i], s)
    #


def print_features(column_names, indices, vector):
    weights = vector[indices]
    if len(indices) > 10:
        items = []
        for i, (idx, weight) in enumerate(zip(indices, weights)):
            feature_name = column_names[idx] if idx < len(column_names) else f"Feature_{idx}"
            item = f"{feature_name}"
            items.append(item)
        head = "{0:.2f} ~ {1:.2f}: ".format(weights[0], weights[-1])
        print(head, items)
    else:
        items = []
        for i, (idx, weight) in enumerate(zip(indices, weights)):
            feature_name = column_names[idx] if idx < len(column_names) else f"Feature_{idx}"
            item = f"({feature_name}: {weight:.2f})"
            items.append(item)
        print(", ".join(items))




def recover_voca_ngram(new_voca):
    row_names = []
    for item in new_voca:
        e = item
        try:
            if item[0] == "(":
                e = ast.literal_eval(item)
                e = " ".join(e)
        except (ValueError, SyntaxError, TypeError):
            pass
        row_names.append(e)
    return row_names


def main():
    # Load data
    print("Loading data...")
    new_voca, score_mat_new, valid_sb_list = load_voca_matrix_both(10, False)

    med = np.median(score_mat_new, axis=1, keepdims=True)
    mat = score_mat_new - med
    print(f"Data shape: {mat.shape}")

    row_names = recover_voca_ngram(new_voca)

    n_comp = 2
    model = PCA(n_components=n_comp)
    W = model.fit_transform(mat)
    H = model.components_
    # W = W * -1
    # H = H * -1
    # for n_comp in [1, 3, 5, 10]:
    #     run_nmf_eval(model, mat, n_comp)
    run_show_pca(W, H, row_names, valid_sb_list)


if __name__ == '__main__':
    main()
