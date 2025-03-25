import json
import logging
import os
from collections import Counter

import fire
import numpy as np
from sklearn.cluster import KMeans

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import init_logging
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_rp_path

LOG = logging.getLogger(__name__)


def load_ngrams(sb):
    save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "ngram_93_all_sel", f"{sb}.json")
    j = json.load(open(save_path))
    pre = "<reason>"
    post = "</reason>"
    output = []
    for sent in j:
        if pre not in sent or post not in sent:
            continue

        st = sent.find(pre) + len(pre)
        ed = sent.find(post)
        text = sent[st:ed]
        output.append(text)
    return output




def print_clusters(clusters, n_show=20):
    for cluster_id in clusters:
        print(f"Cluster {cluster_id} has {len(clusters[cluster_id])} members:")
        for i, sub_text in enumerate(clusters[cluster_id][:n_show]):  # Print first 10 for brevity
            print(f"  {i + 1}. {sub_text}")

        if len(clusters[cluster_id]) > n_show:
            print(f"  ... and {len(clusters[cluster_id]) - n_show} more")
        print()


import numpy as np
from scipy.spatial.distance import euclidean


def calculate_cluster_distances(embeddings, cluster_model):
    # Get cluster centers and labels
    centers = cluster_model.cluster_centers_
    labels = cluster_model.labels_

    # Initialize dictionaries to store distances for each cluster
    max_distances = {}
    avg_distances = {}
    counts = {}

    # Initialize data for each cluster
    for i in range(len(centers)):
        max_distances[i] = 0
        avg_distances[i] = 0
        counts[i] = 0

    # Calculate distances from each point to its cluster center
    for i, point in enumerate(embeddings):
        cluster_idx = labels[i]
        center = centers[cluster_idx]

        distance = euclidean(point, center)

        # Update max distance if needed
        max_distances[cluster_idx] = max(max_distances[cluster_idx], distance)

        # Add to total distance (for averaging later)
        avg_distances[cluster_idx] += distance
        counts[cluster_idx] += 1

    # Calculate averages
    for i in range(len(centers)):
        if counts[i] > 0:
            avg_distances[i] /= counts[i]

    return {
        'max_distances': max_distances,
        'avg_distances': avg_distances,
        'counts': counts
    }

def main(sb="TwoXChromosomes"):
    model_name = f"bert_ts_{sb}"
    init_logging()
    sub_text_path = get_rp_path("ngram_93_all_sub_sel", f"{sb}.json")
    j = json.load(open(sub_text_path))
    text_list = [e["sub_text"] for e in j]
    if len(text_list) < 8:
        print("Skip clustering")
        return

    LOG.info("Loading library")
    from rule_gen.reddit.base_bert.extract_embeddings import BertHiddenStatesExtractor
    LOG.info("Loading model")
    bert = BertHiddenStatesExtractor(model_name)

    def get_emb(t):
        embs = bert.get_sentence_embedding(t, [1, 4, 7, 11], 'mean')
        return np.mean(embs, axis=0)[0]

    embs = list(map(get_emb, text_list))

    LOG.info(f"Running Clustering ... ")
    # eps = 0.1
    # cluster_model = DBSCAN(eps, metric="cosine").fit(embs)
    # n_clusters = max(cluster_model.labels_) + 1
    # LOG.info(f"{n_clusters} clusters found ")
    # labels = cluster_model.labels_  # These are the cluster assignments for each point

    n_clusters = min(len(text_list) // 8, 20)
    cluster_model = KMeans(n_clusters=n_clusters, random_state=0)
    cluster_model.fit(embs)
    labels = cluster_model.labels_
    results = calculate_cluster_distances(embs, cluster_model)

    print(Counter(labels))
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = {"text_list": []}
        clusters[label]["text_list"].append(text_list[idx])
        clusters[label]["max_distances"] = results['max_distances'][label]
        clusters[label]["avg_distances"] = results['avg_distances'][label]

    cluster_path = get_rp_path("ngram_93_all_sel_cluster", f"{sb}.json")

    cluster_data = []
    for cluster_id in clusters:
        cluster_data.append(clusters[cluster_id])
    cluster_data.sort(key=lambda x: x['avg_distances'])

    with open(cluster_path, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=4)
    # print_clusters(clusters, 20)


if __name__ == "__main__":
    fire.Fire(main)
