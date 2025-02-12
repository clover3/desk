import fire
import numpy as np
import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn_extra.cluster import KMedoids
from sklearn.cluster import KMeans
from chair.list_lib import right, left
from chair.misc_lib import make_parent_exists, SuccessCounter
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.corpus_sim.compute_sim import load_pickle_from
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.text_gen.knn_gen import generate_clusters_from_knn

def show_logistic_regression_perf(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    train_pred = clf.predict(X_train)

    train_score = f1_score(y_train, train_pred)
    val_pred = clf.predict(X_test)
    val_score = f1_score(y_test, val_pred)
    print(f"Training F1 Score: {train_score:.4f}")
    print(f"Validation F1 Score: {val_score:.4f}")


def get_logistic_regression_weights(X, y):
    """
    Perform logistic regression and return the model weights.

    Parameters:
    X (array-like): Feature matrix of shape (n_samples, n_features)
    y (array-like): Target vector of shape (n_samples,)

    Returns:
    dict: Dictionary containing intercept and feature coefficients
    """
    # Initialize and fit the model

    model = LogisticRegression(random_state=42)
    model.fit(X, y)
    return model.coef_[0]



def main(sb = "history"):
    save_path = get_reddit_train_data_path_ex("train_data2", sb, "train")
    items = read_csv(save_path)
    n_item = 10000
    items = items[:n_item]
    labels = np.array(list(map(int, right(items))))
    texts = left(items)

    save_path = os.path.join(output_root_path, "reddit", "pickles", f"bert2_{sb}.pkl")
    file_name = f"bert2_{sb}.json"

    embeddings = load_pickle_from(save_path)
    embeddings = embeddings[:n_item]
    B, L, D = embeddings.shape
    # embeddings = np.reshape(embeddings, [B, L * D])
    embeddings = embeddings[:, 0, :]
    rep_pos = [embeddings[i] for i in range(len(embeddings)) if labels[i]]
    pos_file_name = f"bert2_{sb}.json"


    rep_neg = [embeddings[i] for i in range(len(embeddings)) if not labels[i]]
    neg_file_name = f"bert2_{sb}_neg.json"
    todo =  [(rep_pos, pos_file_name), (rep_neg, neg_file_name)]
    for rep , file_name in todo:
        n_clusters = 30
        kmedoids = KMeans(n_clusters=n_clusters, random_state=0).fit(rep)
        k_cents = kmedoids.cluster_centers_

        clusters = generate_clusters_from_knn(
            embeddings, labels, texts,
            k_cents, [1 for _ in k_cents], ["" for _ in k_cents], k=5)

        text_clusters = []
        sc = SuccessCounter()
        for i, cluster in enumerate(clusters):
            for i in cluster.orig_neighbor_indices:
                sc.add(labels[i] == 1)
            info = cluster.get_info()
            mean_dist = np.mean(info["sample_distances"])
            print(info["sample_distances"])
            text_clusters.append((mean_dist, info['sample_texts']))
        text_clusters.sort()
        text_clusters = right(text_clusters)
        print(sc.get_suc_prob())

        cluster_path = os.path.join(output_root_path, "clusters", file_name)
        make_parent_exists(cluster_path)
        with open(cluster_path, "w") as f:
            json.dump(text_clusters, f, indent=True)


if __name__ == "__main__":
    fire.Fire(main)