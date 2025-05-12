from collections import Counter
from scipy.sparse import csr_matrix
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import normalize
from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.term_norm_match import load_doc_id_to_bow, \
    load_doc_id_to_response


class BowClusterer:
    def __init__(self, n_clusters=5, min_term_freq=2):
        """
        Initialize the BoW clusterer.

        Args:
            n_clusters (int): Number of clusters for KMeans
            min_term_freq (int): Minimum frequency for a term to be included in vocabulary
        """
        self.n_clusters = n_clusters
        self.min_term_freq = min_term_freq
        self.vocabulary = {}
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)

    def build_vocabulary(self, counters):
        """
        Build vocabulary from all counters, filtering out rare terms.

        Args:
            counters (list): List of Counter objects
        """
        # Count total frequency of each term across all documents
        total_freq = Counter()
        for counter in counters:
            total_freq.update(counter)

        # Filter terms by minimum frequency
        filtered_vocab = {term for term, freq in total_freq.items()
                          if freq >= self.min_term_freq}

        print(f"Vocabulary size before filtering: {len(total_freq)}")
        print(f"Vocabulary size after filtering (min_freq={self.min_term_freq}): {len(filtered_vocab)}")

        # Create word to index mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(filtered_vocab))}
        self.inv_vocabulary = {idx: word for word, idx in self.vocabulary.items()}

    def counters_to_matrix(self, counters):
        """
        Convert list of Counter objects to sparse matrix.

        Args:
            counters (list): List of Counter objects

        Returns:
            sparse matrix: Document-term matrix
        """
        data = []
        rows = []
        cols = []

        for doc_idx, counter in enumerate(counters):
            for word, count in counter.items():
                if word in self.vocabulary:
                    data.append(count)
                    rows.append(doc_idx)
                    cols.append(self.vocabulary[word])

        matrix = csr_matrix((data, (rows, cols)),
                            shape=(len(counters), len(self.vocabulary)))
        return matrix

    def fit_predict(self, counters, normalize_features=True):
        """
        Fit KMeans and predict clusters.

        Args:
            counters (list): List of Counter objects
            normalize_features (bool): Whether to normalize features

        Returns:
            array: Cluster labels
        """
        # Build vocabulary and convert to matrix
        self.build_vocabulary(counters)
        X = self.counters_to_matrix(counters)

        # Normalize if requested (useful for documents of different lengths)
        if normalize_features:
            X = normalize(X, norm='l2')

        # Fit and predict
        self.labels_ = self.kmeans.fit_predict(X)
        self.X = X  # Store for later use

        return self.labels_

    def get_top_words_per_cluster(self, n_words=10):
        top_words = {}

        for cluster_idx in range(self.n_clusters):
            center = self.kmeans.cluster_centers_[cluster_idx]
            top_indices = center.argsort()[-n_words:][::-1]
            top_words[cluster_idx] = [self.inv_vocabulary[idx] for idx in top_indices]

        return top_words



from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram, linkage
import pandas as pd


class AdvancedBowClusterer(BowClusterer):
    def __init__(self, n_clusters=5, min_term_freq=2, use_tfidf=True,
                 clustering_method='kmeans', max_features=None):
        super().__init__(n_clusters, min_term_freq)
        self.use_tfidf = use_tfidf
        self.clustering_method = clustering_method
        self.max_features = max_features

        if clustering_method == 'hierarchical':
            self.clusterer = AgglomerativeClustering(n_clusters=n_clusters)
        else:
            self.clusterer = self.kmeans

    def build_vocabulary(self, counters):
        total_freq = Counter()
        for counter in counters:
            total_freq.update(counter)

        # Filter terms by minimum frequency
        filtered_terms = [(term, freq) for term, freq in total_freq.items()
                          if freq >= self.min_term_freq]

        # If max_features is set, keep only the most frequent terms
        if self.max_features and len(filtered_terms) > self.max_features:
            filtered_terms.sort(key=lambda x: x[1], reverse=True)
            filtered_terms = filtered_terms[:self.max_features]
            print(f"Limited to top {self.max_features} features")

        filtered_vocab = {term for term, _ in filtered_terms}

        print(f"Vocabulary size before filtering: {len(total_freq)}")
        print(f"Vocabulary size after filtering (min_freq={self.min_term_freq}): {len(filtered_vocab)}")

        # Create word to index mapping
        self.vocabulary = {word: idx for idx, word in enumerate(sorted(filtered_vocab))}
        self.inv_vocabulary = {idx: word for word, idx in self.vocabulary.items()}

    def fit_predict(self, counters, normalize_features=True):
        self.build_vocabulary(counters)
        X = self.counters_to_matrix(counters)

        if self.use_tfidf:
            tfidf = TfidfTransformer()
            X = tfidf.fit_transform(X)

        if normalize_features:
            X = normalize(X, norm='l2')

        self.X = X
        if self.clustering_method == 'hierarchical':
            self.labels_ = self.clusterer.fit_predict(X.toarray())
        else:
            self.labels_ = self.clusterer.fit_predict(X)

        return self.labels_

    def get_top_words_per_cluster(self, n_words=10):
        """
        Get top words per cluster for both KMeans and Hierarchical clustering.

        Args:
            n_words (int): Number of top words to return per cluster

        Returns:
            dict: Dictionary mapping cluster IDs to lists of top words
        """
        top_words = {}

        if self.clustering_method == 'kmeans':
            # For KMeans, use cluster centers
            for cluster_idx in range(self.n_clusters):
                center = self.clusterer.cluster_centers_[cluster_idx]
                top_indices = center.argsort()[-n_words:][::-1]
                top_words[cluster_idx] = [self.inv_vocabulary[idx] for idx in top_indices]
        else:
            # For hierarchical clustering, compute centroid from actual documents
            for cluster_idx in range(self.n_clusters):
                # Get documents in this cluster
                cluster_mask = self.labels_ == cluster_idx

                if cluster_mask.sum() == 0:
                    top_words[cluster_idx] = []
                    continue

                # Calculate mean of cluster documents
                cluster_docs = self.X[cluster_mask]
                center = cluster_docs.mean(axis=0).A1  # Convert to 1D array

                # Get top words
                top_indices = center.argsort()[-n_words:][::-1]
                top_words[cluster_idx] = [self.inv_vocabulary[idx] for idx in top_indices]

        return top_words

    def get_cluster_summary(self, documents, n_words=5):
        top_words = self.get_top_words_per_cluster_advanced(n_words)

        summary_data = []
        for cluster_id in range(self.n_clusters):
            mask = self.labels_ == cluster_id
            cluster_docs = [doc for doc, m in zip(documents, mask) if m]

            summary_data.append({
                'Cluster': cluster_id,
                'Size': len(cluster_docs),
                'Top Words': ', '.join(top_words[cluster_id]),
                'Sample Docs': cluster_docs[:2]  # First 2 documents
            })

        return pd.DataFrame(summary_data)

    def get_distinguishing_words_per_cluster(self, n_words=10, uniqueness_weight=0.5):
        """
        Get words that best distinguish each cluster from others.
        Uses TF-IDF like scoring to find words that are important in a cluster
        but not common across all clusters.

        Args:
            n_words (int): Number of distinguishing words per cluster
            uniqueness_weight (float): Balance between frequency and uniqueness (0-1)

        Returns:
            dict: Dictionary mapping cluster IDs to lists of distinguishing words
        """
        import numpy as np

        distinguishing_words = {}

        # Calculate document frequencies across clusters
        cluster_word_freq = {}
        cluster_sizes = {}

        for cluster_idx in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_idx
            cluster_size = cluster_mask.sum()
            cluster_sizes[cluster_idx] = cluster_size

            if cluster_size == 0:
                cluster_word_freq[cluster_idx] = np.zeros(len(self.vocabulary))
                continue

            # Sum word frequencies in this cluster
            cluster_docs = self.X[cluster_mask]
            cluster_freq = np.array(cluster_docs.sum(axis=0)).flatten()
            # Normalize by cluster size
            cluster_word_freq[cluster_idx] = cluster_freq / cluster_size

        # Calculate global word frequencies
        total_docs = len(self.labels_)
        global_freq = np.array(self.X.sum(axis=0)).flatten() / total_docs

        # For each cluster, calculate distinguishing score
        for cluster_idx in range(self.n_clusters):
            if cluster_sizes[cluster_idx] == 0:
                distinguishing_words[cluster_idx] = []
                continue

            # Calculate TF-IDF-like score
            cluster_freq = cluster_word_freq[cluster_idx]

            # Avoid division by zero
            global_freq_safe = np.maximum(global_freq, 1e-10)

            # Score combines frequency in cluster and inverse frequency globally
            tf_score = cluster_freq
            idf_score = np.log(1 + 1.0 / global_freq_safe)

            # Combine scores with weighting
            distinguishing_score = (uniqueness_weight * tf_score * idf_score +
                                    (1 - uniqueness_weight) * tf_score)

            # Get top distinguishing words
            top_indices = distinguishing_score.argsort()[-n_words:][::-1]
            distinguishing_words[cluster_idx] = [self.inv_vocabulary[idx] for idx in top_indices]

        return distinguishing_words

    def get_exclusive_words_per_cluster(self, n_words=10, min_ratio=2.0):
        """
        Get words that appear significantly more in one cluster than others.

        Args:
            n_words (int): Number of exclusive words per cluster
            min_ratio (float): Minimum ratio of cluster frequency to average other cluster frequency

        Returns:
            dict: Dictionary mapping cluster IDs to lists of exclusive words
        """
        import numpy as np

        exclusive_words = {}

        # Calculate normalized word frequencies for each cluster
        cluster_word_freq = {}
        for cluster_idx in range(self.n_clusters):
            cluster_mask = self.labels_ == cluster_idx
            cluster_size = cluster_mask.sum()

            if cluster_size == 0:
                cluster_word_freq[cluster_idx] = np.zeros(len(self.vocabulary))
                continue

            cluster_docs = self.X[cluster_mask]
            # Normalize by number of documents in cluster
            cluster_word_freq[cluster_idx] = np.array(cluster_docs.sum(axis=0)).flatten() / cluster_size

        # For each cluster, find words that are much more common than in other clusters
        for cluster_idx in range(self.n_clusters):
            if cluster_idx not in cluster_word_freq or cluster_word_freq[cluster_idx].sum() == 0:
                exclusive_words[cluster_idx] = []
                continue

            current_freq = cluster_word_freq[cluster_idx]

            # Calculate average frequency in other clusters
            other_freqs = []
            for other_idx in range(self.n_clusters):
                if other_idx != cluster_idx and other_idx in cluster_word_freq:
                    other_freqs.append(cluster_word_freq[other_idx])

            if not other_freqs:
                exclusive_words[cluster_idx] = []
                continue

            avg_other_freq = np.mean(other_freqs, axis=0)

            # Avoid division by zero
            avg_other_freq_safe = np.maximum(avg_other_freq, 1e-10)

            # Calculate ratio
            ratios = current_freq / avg_other_freq_safe

            # Filter words that meet minimum ratio requirement
            exclusive_indices = np.where(ratios >= min_ratio)[0]

            # Sort by ratio and get top words
            if len(exclusive_indices) > 0:
                sorted_exclusive = exclusive_indices[np.argsort(ratios[exclusive_indices])[::-1]]
                top_exclusive = sorted_exclusive[:n_words]
                exclusive_words[cluster_idx] = [self.inv_vocabulary[idx] for idx in top_exclusive]
            else:
                exclusive_words[cluster_idx] = []

        return exclusive_words


# Example usage
if __name__ == "__main__":
    # Create example data: list of Counter objects
    print("Loading data")
    d: dict[str, Counter[tuple, int]] = load_doc_id_to_bow()
    doc_id_to_res = load_doc_id_to_response()
    keys = list(d.keys())
    counters = [d[k] for k in keys]

    # First, let's see the term statistics
    print("\nRunning hierarchical clustering with filtered vocabulary")
    clusterer = AdvancedBowClusterer(
        n_clusters=10,
        min_term_freq=5,  # Only include terms that appear at least 5 times
        use_tfidf=False,
        clustering_method='hierarchical',
        max_features=5000  # Optional: limit to top 5000 features
    )
    labels = clusterer.fit_predict(counters)

    print("\nTop words per cluster:")
    top_words = clusterer.get_top_words_per_cluster(n_words=5)
    for cluster, words in top_words.items():
        print(f"Cluster {cluster}: {words}")

    print("\nDistinguishing words per cluster (TF-IDF style):")
    distinguishing_words = clusterer.get_distinguishing_words_per_cluster(n_words=5, uniqueness_weight=0.7)
    for cluster, words in distinguishing_words.items():
        print(f"Cluster {cluster}: {words}")

    print("\nExclusive words per cluster (appear much more in this cluster):")
    exclusive_words = clusterer.get_exclusive_words_per_cluster(n_words=5, min_ratio=3.0)
    for cluster, words in exclusive_words.items():
        print(f"Cluster {cluster}: {words}")

    # Calculate silhouette score
    silhouette_avg = silhouette_score(clusterer.X, labels)
    print(f"\nSilhouette Score: {silhouette_avg:.3f}")
