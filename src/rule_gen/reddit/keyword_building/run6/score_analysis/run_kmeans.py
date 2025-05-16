from collections import Counter

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_analysis.common import load_run_score_matrix
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import correlation


def run_clustering(df, k, terms, distance_threshold=1.5):
    # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(df)
    X_scaled = df.values

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_

    # Calculate distances from each point to its assigned centroid
    distances = np.zeros(len(df))
    for i in range(len(df)):
        cluster_label = labels[i]
        distances[i] = correlation(X_scaled[i], centroids[cluster_label])

    # Create mask for points that are within the threshold distance
    mask = distances <= distance_threshold

    # Add cluster labels, terms and distances to original data
    df = df.copy()  # Avoid modifying original DataFrame
    df['cluster'] = labels
    df['terms'] = terms
    df['distance'] = distances

    # Filter out points that exceed the distance threshold
    df_filtered = df[mask].copy()

    # Set cluster to -1 for outliers in the original dataframe
    df.loc[~mask, 'cluster'] = -1

    # Count cluster sizes (excluding outliers)
    cluster_counts = Counter(df_filtered['cluster'])
    print(f"Distance threshold: {distance_threshold}")
    print(f"Points retained: {sum(mask)}/{len(mask)} ({sum(mask) / len(mask) * 100:.1f}%)")
    print("Cluster sizes:", {k: v for k, v in sorted(cluster_counts.items()) if k != -1})

    # Get column names (excluding new columns)
    feature_columns = df.columns.difference(['cluster', 'terms', 'distance'])

    mean_val = []
    for cluster_num in range(k):
        # Rows in the cluster (from filtered dataset)
        cluster_rows = df_filtered[df_filtered['cluster'] == cluster_num]
        if len(cluster_rows) == 0:
            mean_val.append(-np.inf)  # Empty clusters will be sorted to the end
            continue
        centroid = kmeans.cluster_centers_[cluster_num]
        mean_val.append(centroid.mean())

    print_indices = np.argsort(mean_val)[::-1]
    for cluster_num in print_indices[:20]:
        if mean_val[cluster_num] == -np.inf:
            continue  # Skip empty clusters

        print(f"\nCluster {cluster_num}:")

        # Rows in the cluster (from filtered dataset)
        cluster_rows = df_filtered[df_filtered['cluster'] == cluster_num]
        if len(cluster_rows) == 0:
            print("Empty cluster after filtering")
            continue

        centroid = kmeans.cluster_centers_[cluster_num]

        print(f"Avg val: {np.mean(centroid):.3f}, Size: {len(cluster_rows)}")
        influential_indices = np.argsort(-np.abs(centroid))  # Descending order
        influential_columns = [feature_columns[i] for i in influential_indices[:5]]

        print("Most influential features (in order):")
        s_list = []
        for col in influential_columns:
            idx = df.columns.get_loc(col)
            s = f"{col}: {centroid[idx]:.3f}"
            s_list.append(s)
        print(" ".join(s_list))

        # Display top 5 rows with important columns + 'terms', 'cluster', and 'distance'
        display_columns = influential_columns + ['terms', 'cluster', 'distance']
        print(cluster_rows[display_columns].sort_values('distance').head())
        print(" / ".join(cluster_rows["terms"][:200]))


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
    # X = score_mat
    # X_norm = (X - X.mean(axis=1, keepdims=True)) / X.std(axis=1, keepdims=True)
    # correlations = X_norm @ X_norm
    # return correlations, term_list


if __name__ == '__main__':
    n_list = list(range(1, 5))
    score_mat, term_list, valid_sb_list = load_mat_terms(n_list)
    df = pd.DataFrame(score_mat, columns=valid_sb_list)
    print(df.shape)
    distance_threshold = 0.2
    run_clustering(df, 100, term_list, distance_threshold)
