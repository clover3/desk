import json
import os

from chair.misc_lib import group_by, make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.knn import SentenceEncoder
from rule_gen.open_ai_mod.train_proto import get_data_arguments
from rule_gen.reddit.proto.train_proto_reddit import apply_tokenize, get_tokenize_formatter
from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg
import numpy as np
from sklearn_extra.cluster import KMedoids



if __name__ == "__main__":

    # Setup and data preparation
    encoder = SentenceEncoder()
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(False)
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)

    # Set random seed for reproducibility

    tokenize_format = get_tokenize_formatter(encoder.tokenizer, dataset_args.max_length)
    tokenized_train, _tokenized_eval = apply_tokenize(train_dataset, eval_dataset, tokenize_format)
    embeddings, labels = encoder.encode_and_label(tokenized_train)
    texts = [item['text'] for item in train_dataset]

    rep_pos = [embeddings[i] for i in range(len(embeddings)) if labels[i]]
    n_clusters = 10
    kmedoids = KMedoids(n_clusters=n_clusters, random_state=0).fit(rep_pos)
    k_cents = kmedoids.cluster_centers_

    groups = group_by(range(len(rep_pos)), lambda idx: kmedoids.labels_[idx])
    clustered = []
    for cluster_idx in range(n_clusters):
        entries = groups[cluster_idx]
        clustered.append([texts[i] for i in entries])

    cluster_path = os.path.join(output_root_path, "clusters", "KMedoids.json")
    make_parent_exists(cluster_path)
    with open(cluster_path, "w") as f:
        json.dump(clustered, f)
