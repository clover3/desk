import dataclasses
from typing import List, Set, Dict, Any, Tuple
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler


@dataclasses.dataclass
class Cluster:
    src_text: str
    src_label: int
    knn_texts: list[str]
    knn_labels: list[int]
    distances: np.ndarray
    embeddings: np.ndarray

    @property
    def cluster_labels(self) -> list[int]:
        """Get all labels in the cluster including source label."""
        return self.knn_labels

    @property
    def cluster_texts(self) -> list[str]:
        """Get all texts in the cluster including source text."""
        return self.knn_texts

    @property
    def size(self) -> int:
        """Get the size of the cluster."""
        return len(self.knn_texts)

    @property
    def neighbor_indices(self) -> Set[int]:
        """Get indices of neighbors based on the order in knn_texts."""
        return set(range(len(self.knn_texts)))

    @property
    def label_consistency(self) -> float:
        """Calculate the proportion of labels that match the majority label."""
        label_counts = np.bincount(self.cluster_labels)
        majority_label = np.argmax(label_counts)
        return float(label_counts[majority_label] / len(self.cluster_labels))

    @property
    def majority_label(self) -> int:
        """Get the most common label in the cluster."""
        return int(np.argmax(np.bincount(self.cluster_labels)))

    def predict_weighted(self) -> Tuple[int, float]:
        """
        Make a prediction using inverse distance weighting.
        Returns (predicted_label, confidence)
        """
        weights = 1.0 / (self.distances + 1e-6)
        weights = weights / np.sum(weights)

        unique_labels = np.unique(self.cluster_labels)
        label_probs = {}

        for label in unique_labels:
            label_mask = np.array(self.cluster_labels) == label
            label_weight = np.sum(weights[label_mask])
            label_probs[label] = label_weight

        predicted_label = max(label_probs.items(), key=lambda x: x[1])[0]
        confidence = label_probs[predicted_label]

        return predicted_label, confidence

    @property
    def unique_labels(self) -> Set[int]:
        """Get all unique labels in the cluster."""
        return set(self.cluster_labels)

    def overlaps_with(self, other: 'Cluster') -> bool:
        """Check if this cluster overlaps with another cluster."""
        return len(self.neighbor_indices.intersection(other.neighbor_indices)) > 0

    def get_score(self) -> float:
        """Calculate cluster score based on label consistency and distances."""
        pred_label, confidence = self.predict_weighted()
        label_score = 0.7 * self.label_consistency + 0.3 * confidence

        cluster_points = self.embeddings[list(self.neighbor_indices)]
        pairwise_distances = np.linalg.norm(
            cluster_points[:, np.newaxis] - cluster_points, axis=2
        )
        distance_penalty = 0.1 * np.mean(pairwise_distances)

        return label_score - distance_penalty

    def get_info(self, max_samples: int = 5) -> Dict[str, Any]:
        """Get a dictionary containing cluster information."""
        pred_label, confidence = self.predict_weighted()
        return {
            'size': self.size,
            'majority_label': self.majority_label,
            'weighted_prediction': pred_label,
            'prediction_confidence': confidence,
            'label_consistency': self.label_consistency,
            'src_text': self.src_text,
            'src_label': self.src_label,
            'sample_texts': self.cluster_texts[:max_samples],
            'sample_labels': self.cluster_labels[:max_samples],
            'sample_distances': self.distances[:max_samples].tolist(),
            'total_unique_labels': len(self.unique_labels)
        }


def generate_clusters_from_knn(
        X_train: np.ndarray,
        y_train: np.ndarray,
        texts_train: List[str],
        X_val: np.ndarray,
        y_val: np.ndarray,
        texts_val: List[str],
        k: int = 3) -> List[Cluster]:
    """Generate clusters based on k-nearest neighbors from validation set points."""
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    nbrs = NearestNeighbors(n_neighbors=k, n_jobs=-1)
    nbrs.fit(X_train_scaled)

    distances, indices = nbrs.kneighbors(X_val_scaled)

    clusters = []
    for i, (neighbor_indices, neighbor_distances) in enumerate(zip(indices, distances)):
        # Get the texts and labels for the k nearest neighbors
        knn_texts = [texts_train[idx] for idx in neighbor_indices]
        knn_labels = [y_train[idx] for idx in neighbor_indices]

        cluster = Cluster(
            src_text=texts_val[i],
            src_label=y_val[i],
            knn_texts=knn_texts,
            knn_labels=knn_labels,
            distances=neighbor_distances,
            embeddings=X_train_scaled
        )
        clusters.append(cluster)
    return clusters



def filter_clusters(clusters: List[Cluster]) -> List[Cluster]:
    # Filter and sort clusters by score
    clusters.sort(key=lambda x: x.get_score(), reverse=True)
    # Remove overlapping clusters
    final_clusters = []
    used_indices = set()

    for cluster in clusters:
        if not any(cluster.overlaps_with(selected) for selected in final_clusters):
            if not cluster.neighbor_indices.intersection(used_indices):
                final_clusters.append(cluster)
                used_indices.update(cluster.neighbor_indices)

    return final_clusters


# Example usage:
if __name__ == "__main__":
    from rule_gen.open_ai_mod.knn import SentenceEncoder
    from rule_gen.open_ai_mod.train_proto import get_data_arguments
    from rule_gen.reddit.proto.train_proto_reddit import apply_tokenize, get_tokenize_formatter
    from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg

    # Setup and data preparation
    encoder = SentenceEncoder()
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(False)
    full_dataset, _ = get_datasets_from_dataset_arg(dataset_builder, dataset_args)

    # Set random seed for reproducibility
    np.random.seed(42)

    # Calculate sizes
    total_size = len(full_dataset)
    n_eval_size = 10
    train_size = total_size - n_eval_size

    # Create train/eval split using Dataset's select method
    indices = list(range(total_size))  # Create list of indices
    np.random.shuffle(indices)  # Shuffle indices

    # Use select method which is compatible with HuggingFace Datasets
    train_dataset = full_dataset.select(indices[:train_size])
    eval_dataset = full_dataset.select(indices[train_size:])


    tokenize_format = get_tokenize_formatter(encoder.tokenizer, dataset_args.max_length)
    tokenized_train, tokenized_eval = apply_tokenize(train_dataset, eval_dataset, tokenize_format)
    embeddings, labels = encoder.encode_and_label(tokenized_train)
    embeddings_eval, labels_eval = encoder.encode_and_label(tokenized_eval)
    texts = [item['text'] for item in train_dataset]
    texts_eval = [item['text'] for item in eval_dataset]

    # Generate clusters
    clusters = generate_clusters_from_knn(
        embeddings, labels, texts,
        embeddings_eval, labels_eval, texts_eval, k=3)

    # Print cluster information
    correct_majority = []
    correct_weighted = []

    for i, cluster in enumerate(clusters):
        info = cluster.get_info()
        print(f"\nCluster {i}:")
        print(f"Size: {info['size']}")
        print(f"Source text: {info['src_text']}")
        print(f"Source label: {info['src_label']}")
        print(f"Majority Label: {info['majority_label']}")
        print(f"Weighted Prediction: {info['weighted_prediction']} (confidence: {info['prediction_confidence']:.3f})")

        correct_majority.append(info['majority_label'] == info['src_label'])
        correct_weighted.append(info['weighted_prediction'] == info['src_label'])

        print(f"Label Consistency: {info['label_consistency']:.2f}")
        print(f"Number of Unique Labels: {info['total_unique_labels']}")
        print("\nSample texts with distances:")
        for text, label, dist in zip(info['sample_texts'],
                                     info['sample_labels'],
                                     info['sample_distances']):
            print(f"[Label {label}, Distance {dist:.3f}] {text[:100]}...")
        print("-" * 80)

    print(f"Majority Vote Accuracy: {np.mean(correct_majority):.2f}")
    print(f"Weighted Prediction Accuracy: {np.mean(correct_weighted):.2f}")