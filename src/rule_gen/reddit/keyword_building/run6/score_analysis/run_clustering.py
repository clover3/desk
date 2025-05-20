import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from collections import Counter

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_run_score_matrix


def run_clustering(df, eps, min_samples, terms):
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df)
    X_scaled = df

    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)

    labels = dbscan.fit_predict(X_scaled)

    # Count number of clusters (excluding noise points which are labeled -1)
    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
    print(f"Number of clusters found: {n_clusters}")
    print(f"Number of noise points: {list(labels).count(-1)}")

    # Cluster distribution
    cluster_counts = Counter(labels)
    print("Cluster sizes:", {k: v for k, v in sorted(cluster_counts.items()) if k != -1})

    # Add cluster labels and terms to original data
    df = df.copy()  # Avoid modifying original DataFrame
    df['cluster'] = labels
    df['terms'] = terms

    # Get column names (excluding new columns)
    feature_columns = df.columns.difference(['cluster', 'terms'])

    # For each cluster (excluding noise points)
    for cluster_num in sorted(set(labels)):
        if cluster_num == -1:
            continue  # Skip noise points

        print(f"\nCluster {cluster_num}:")

        # Rows in the cluster
        cluster_rows = df[df['cluster'] == cluster_num]

        # Calculate cluster centroid
        centroid = X_scaled[labels == cluster_num].mean(axis=0)
        # centroid_raw = df[labels == cluster_num].mean(axis=0)

        # Influential columns (based on centroid magnitude)
        influential_indices = np.argsort(-np.abs(centroid))  # Descending order
        influential_columns = [feature_columns[i] for i in influential_indices[:5]]

        print(f"Size: {len(cluster_rows)} points")
        print("Most influential features (in order):")
        for col in influential_columns:
            idx = feature_columns.get_loc(col)
            print(f"  {col} (scaled importance: {centroid[idx]:.3f})")

        # Display top 5 rows with important columns + 'terms' and 'cluster'
        display_columns = influential_columns + ['terms', 'cluster']
        print(cluster_rows[display_columns].head())

    # Show information about noise points if any exist
    if -1 in labels:
        print("\nNoise points (not assigned to any cluster):")
        noise_rows = df[df['cluster'] == -1]
        print(f"Number of noise points: {len(noise_rows)}")
        print(noise_rows[['terms', 'cluster']].head())


def load_mat_terms(n_list) -> tuple[np.array, list[str], list[str]]:
    print("Loading data...")
    dir_name = "run6_10k_score"
    sb_len = None
    score_mat_list = []
    term_list_ex = []
    for n in n_list:
        score_mat, valid_sb_list = load_run_score_matrix(dir_name, n)
        score_mat_list.append(score_mat)
        if sb_len is None:
            sb_len = len(valid_sb_list)
        else:
            assert sb_len == len(valid_sb_list)
        term_list = load_run6_10k_text(n)
        term_list_ex.extend(term_list)
    score_mat = np.concatenate(score_mat_list, axis=0)
    term_list = term_list_ex
    return score_mat, term_list, valid_sb_list


if __name__ == '__main__':
    n_list = list(range(1, 5))
    score_mat, term_list, valid_sb_list = load_mat_terms(n_list)
    df = pd.DataFrame(score_mat, columns=valid_sb_list)
    print(df.shape)

    # DBSCAN parameters - you may need to tune these
    eps = 2  # Maximum distance between two samples to be considered in the same neighborhood
    min_samples = 5  # Minimum number of samples in a neighborhood for a point to be a core point

    run_clustering(df, eps, min_samples, terms=term_list)