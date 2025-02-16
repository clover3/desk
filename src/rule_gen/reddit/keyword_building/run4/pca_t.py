import pickle
import os
import pickle

import numpy as np

from desk_util.io_helper import read_csv
# import seaborn as sns
# import matplotlib.pyplot as plt
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.run4.pca import analyze_pca_components, load_key
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list


def main():
    # Load data
    print("Loading data...")
    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    X = pickle.load(open(feature_save_path, "rb"))
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)

    subreddit_list = get_split_subreddit_list("train")

    print(f"Data shape: {X.shape}")

    # 10000, 60

    # Analyze PCA components
    results = analyze_pca_components(X, n_components=20)
    d = load_key()

    # Print explained variance for each component
    print("\nExplained variance ratio per component:")
    for i, var in enumerate(results['explained_variance_ratio']):
        print(f"PC{i + 1}: {var:.3f} (cumulative: {np.sum(results['explained_variance_ratio'][:i + 1]):.3f})")

    X_transformed = results["X_transformed"]
    print("X_transformed", X_transformed.shape)
    for i in range(len(X_transformed)):
        top_indices = np.argsort(np.abs(X_transformed[i]))[-5:][::-1]
        top_s = ", ".join(["{0} ({1:.2f})".format(j, X_transformed[i, j]) for j in top_indices])
        print(subreddit_list[i], top_s)

    # Print top contributing features for each component
    print("\nTop contributing features per component:")
    for pc, features in results['top_features'].items():
        print(f"\n{pc}:")
        for idx, coef in zip(features['indices'], features['coefficients']):
            print(f"Feature {idx}: coefficient = {coef:.3f}")
            pos, neg = [], []
            for sb_idx, v in enumerate(X[idx]):
                sb = subreddit_list[sb_idx]
                if v:
                    pos.append(sb)
                else:
                    neg.append(sb)

            print(items[idx], d[items[idx][0]])
            print("Pos", pos)
            print("Neg", neg)

    # Plot components heatmap
    # plot_components_heatmap(results['components'])
    # plt.show()

    # For a specific feature j, you can see its contribution to each PC
    # by looking at row j of the components matrix
    feature_idx = 0  # Change this to analyze different features
    print(f"\nContributions of feature {feature_idx} to each PC:")
    feature_contributions = results['components'].iloc[feature_idx]
    for pc, contribution in feature_contributions.items():
        print(f"{pc}: {contribution:.3f}")


if __name__ == "__main__":
    main()
