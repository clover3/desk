import logging
import os

import fire
import numpy as np
from chair.list_lib import lflatten
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import init_logging
from rule_gen.cpath import output_root_path
from rule_gen.reddit.bert_pat.ngram_common import load_ngram_outputs, NGramInfo

LOG = logging.getLogger(__name__)

import json


def save_clusters_to_json(clusters, output_file):
    cluster_data = {}
    for cluster_id in clusters:
        cluster_info = {
            "total_members": len(clusters[cluster_id]),
            "members": [
                {"text": ngram.sub_text, "score": round(ngram.score, 4)}
                for ngram in clusters[cluster_id]
            ]
        }
        cluster_data[f"Cluster {cluster_id}"] = cluster_info

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(cluster_data, f, indent=4)


def print_clusters(clusters, n_show=20):
    for cluster_id in clusters:
        print(f"Cluster {cluster_id} has {len(clusters[cluster_id])} members:")
        for i, ngram in enumerate(clusters[cluster_id][:n_show]):  # Print first 10 for brevity
            print(f"  {i + 1}. {ngram.sub_text} (score: {ngram.score:.4f})")

        if len(clusters[cluster_id]) > n_show:
            print(f"  ... and {len(clusters[cluster_id]) - n_show} more")
        print()



def main(sb="TwoXChromosomes"):
    model_name = f"bert2_{sb}"
    init_logging()
    LOG.info("Loading library")
    from sklearn.cluster import KMeans, DBSCAN
    from rule_gen.reddit.base_bert.extract_embeddings import BertHiddenStatesExtractor

    LOG.info("Loading model")
    bert = BertHiddenStatesExtractor(model_name)
    ngram_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93", f"{sb}.json")
    pos, neg = load_ngram_outputs(ngram_path)
    pos: list[list[NGramInfo]] = pos
    pos_reps: list[list[np.ndarray]] = []
    n = len(pos)
    LOG.info(f"Computing embeddings ...")
    for data_idx in range(n):
        # check input id equals
        text = pos[data_idx][0].text
        sub_text_list = [ngram.sub_text for ngram in pos[data_idx]]
        if not sub_text_list:
            continue

        if not isinstance(sub_text_list[0], str):
            continue

        embs_list = bert.get_sub_text_embeddings(text, sub_text_list, layer_no=-1)
        per_item = []
        for embs in embs_list:
            ngram_rep = np.mean(embs, axis=0)
            per_item.append(ngram_rep)
        pos_reps.append(per_item)

    pos_flat: list[NGramInfo] = lflatten(pos)
    pos_reps_flat: list[np.ndarray] = lflatten(pos_reps)

    LOG.info(f"Running Clustering ... ")
    eps = 0.1
    cluster_model = DBSCAN(eps, metric="cosine").fit(pos_reps_flat)
    n_clusters = max(cluster_model.labels_) + 1
    LOG.info(f"{n_clusters} clusters found ")

    labels = cluster_model.labels_  # These are the cluster assignments for each point
    print(labels)
    clusters = {}
    for idx, label in enumerate(labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(pos_flat[idx])


    cluster_save_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_cluster", f"{sb}.json")
    make_parent_exists(cluster_save_path)
    save_clusters_to_json(clusters, cluster_save_path)
    print_clusters(clusters, 20)


if __name__ == "__main__":
    fire.Fire(main)
