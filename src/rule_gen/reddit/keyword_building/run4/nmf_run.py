import numpy as np
import os
import os
import pickle

from sklearn.decomposition import PCA

from desk_util.io_helper import read_csv
# import seaborn as sns
# import matplotlib.pyplot a's plt
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.run4.nmf_eval_common import run_nmf_eval
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list
from rule_gen.reddit.s9.voca.show_pca import run_show_pca


def run_show_nmf(W, H, row_names, column_names, ):
    print("W", W.shape)
    print("H", H.shape)
    n_top_features = 5
    n_components = W.shape[1]
    head = [""] + list(map(str, range(n_components)))
    print("\t".join(head))
    for col_i in range(H.shape[1]):
        s = "\t".join(["{0:.2f}".format(v) for v in H[:, col_i]])
        print(f"{column_names[col_i]}\t{s}")

    for topic_idx, topic in enumerate(H):
        print(f"\nComponent {topic_idx}: Top columns")
        # Get indices of top features for this component
        top_features_idx = topic.argsort()[::-1][:n_top_features]
        top_weights = topic[top_features_idx]

        # Display top features and their weights
        for i, (idx, weight) in enumerate(zip(top_features_idx, top_weights)):
            feature_name = column_names[idx] if idx < len(column_names) else f"Feature_{idx}"
            print(f"  {i + 1}. {feature_name}: {weight:.4f}")

        component_scores = W[:, topic_idx]
        top_item_indices = component_scores.argsort()[::-1][:30]  # Top 10 items

        print(f"\nComponent {topic_idx} - Top Rows:")
        print([(row_names[idx], component_scores[idx]) for idx in top_item_indices])


def main():
    # Load data
    print("Loading data...")
    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    X = pickle.load(open(feature_save_path, "rb"))
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)
    print("Loaded {} texts".format(len(items)))
    subreddit_list = get_split_subreddit_list("train")
    print(f"Data shape: {X.shape}")
    # Xshape = [10000, 60]
    # subreddit_list: 0~ 59
    # items: 0~10000
    n_comp = 1
    model = PCA(n_components=n_comp)
    X = X - np.mean(X, axis=1, keepdims=True)
    W = model.fit_transform(X)
    H = model.components_
    # run_show_pca(W, H, items, subreddit_list)
    run_nmf_eval(model, X, n_comp)


if __name__ == "__main__":
    main()
