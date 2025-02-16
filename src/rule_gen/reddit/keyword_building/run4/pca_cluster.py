import numpy as np
from sklearn.cluster import KMeans

from chair.misc_lib import group_by

44

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
    X_raw = pickle.load(open(feature_save_path, "rb"))
    train_data_path = get_reddit_train_data_path_ex("train_data2", "train_mix", "train")
    items = read_csv(train_data_path)

    subreddit_list = get_split_subreddit_list("train")
    X_T = np.transpose(X_raw, [1, 0])
    X = X_T

    # Analyze PCA components
    # results = analyze_pca_components(X, n_components=20)
    # d = load_key()
    # X_transformed = results["X_transformed"]
    X_transformed = X
    print("X_transformed.shape", X_transformed.shape)
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(X_transformed)
    labels = kmeans.labels_
    print("labels", labels)

    groups = group_by(zip(subreddit_list, labels), lambda x: x[1])
    for c_idx in groups:
        members = ", ".join([sb for sb, _ in groups[c_idx]])
        print(c_idx, members)

    dists = kmeans.transform(X_transformed)
    print("dists", dists.shape)

    for idx, sb in enumerate(subreddit_list):
        top_indices = np.argsort(dists[idx])
        s = " ".join(["{0} ({1:.2f})".format(c_idx, dists[idx][c_idx]) for c_idx in top_indices])
        print(sb, s)




if __name__ == "__main__":
    main()