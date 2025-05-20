import json
from collections import Counter

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans

from scipy.spatial.distance import correlation, sqeuclidean, euclidean

from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms, load_mat_terms_pickled
from rule_gen.reddit.path_helper import get_rp_path


def run_clustering(df, k, terms, terms_text, distance_threshold=1.5):
    # Standardize features
    # scaler = StandardScaler()
    # X_scaled = scaler.fit_transform(df)
    X_scaled = df.values

    # Fit KMeans
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    centroids = kmeans.cluster_centers_
    s = sklearn.metrics.silhouette_score(X_scaled, labels, metric='euclidean')
    print("silhouette_score", s)
    # Calculate distances from each point to its assigned centroid
    distances = np.zeros(len(df))
    for i in range(len(df)):
        cluster_label = labels[i]
        distances[i] = euclidean(X_scaled[i], centroids[cluster_label])

    # sklearn.metrics.silhouette_score(X_scaled, labels, metric='euclidean')
    # Create mask for points that are within the threshold distance
    mask = distances <= distance_threshold

    # Add cluster labels, terms and distances to original data
    df = df.copy()  # Avoid modifying original DataFrame
    df['cluster'] = labels
    df['terms'] = terms
    df["terms_text"] = terms_text
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
    save_data = []
    for cluster_num in print_indices[:10]:
        if mean_val[cluster_num] == -np.inf:
            continue  # Skip empty clusters

        # Rows in the cluster (from filtered dataset)
        cluster_rows = df_filtered[df_filtered['cluster'] == cluster_num]
        if len(cluster_rows) == 0:
            print("Empty cluster after filtering")
            continue

        centroid = kmeans.cluster_centers_[cluster_num]

        # print(f"\nCluster {cluster_num}: Avg val: {np.mean(centroid):.3f}, Size: {len(cluster_rows)}")
        influential_indices = np.argsort(-np.abs(centroid))  # Descending order
        influential_columns = [feature_columns[i] for i in influential_indices[:5]]

        # print("Most influential features (in order):")
        # s_list = []
        # for col in influential_columns:
        #     idx = df.columns.get_loc(col)
        #     s = f"{col}: {centroid[idx]:.3f}"
        #     s_list.append(s)
        # print(" ".join(s_list))

        # Display top 5 rows with important columns + 'terms', 'cluster', and 'distance'
        display_columns = influential_columns + ['terms', 'cluster', 'distance']
        # print(cluster_rows[display_columns].sort_values('distance').head())
        # print(" / ".join(cluster_rows["terms_text"][:200]))

        cluster_info = {
            "cluster_no": cluster_num.tolist(),
            "terms": cluster_rows["terms"].tolist(),
            "average": np.mean(centroid).tolist()
        }
        save_data.append(cluster_info)
    return save_data


if __name__ == '__main__':
    n_list = list(range(1, 10))
    score_mat, term_list, valid_sb_list = load_mat_terms_pickled()
    df = pd.DataFrame(score_mat, columns=valid_sb_list)
    print(df.shape)
    distance_threshold = 1 * 0.1
    term_list_ex = []
    for n in n_list:
        term_text_list = load_run6_10k_text(n)
        term_list_ex.extend(term_text_list)
    term_text_list = term_list_ex

    k = 100
    print("K=", k)
    cluster_output = run_clustering(df, k, term_list, term_text_list, distance_threshold)
    score_path = get_rp_path("clustering", f"{k}_new.json")
    j_str = json.dumps(cluster_output, indent=2)
    # j_str = re.sub(j_str, ",\n[\t]{1,}", ",")
    open(score_path, "w").write(j_str)
