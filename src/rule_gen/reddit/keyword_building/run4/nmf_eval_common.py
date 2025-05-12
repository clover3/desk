import numpy as np
from sklearn.decomposition import NMF
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split


def run_nmf_eval(model, X, n_comp):
    # Split data for evaluation
    X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)
    # Evaluate NMF with different numbers of components
    print(f"\nEvaluating NMF with {n_comp} components...")

    # Train NMF model
    W_train = model.fit_transform(X_train)
    H = model.components_

    # Calculate reconstruction error on training data
    X_train_reconstructed = np.dot(W_train, H)
    train_rmse = np.sqrt(mean_squared_error(X_train, X_train_reconstructed))

    # Calculate reconstruction error on test data
    W_test = model.transform(X_test)
    X_test_reconstructed = np.dot(W_test, H)
    test_rmse = np.sqrt(mean_squared_error(X_test, X_test_reconstructed))

    # Calculate explained variance
    # For NMF, we can use a ratio of reconstruction error to original variance
    total_variance = np.var(X_train) * X_train.size
    unexplained_variance = np.sum((X_train - X_train_reconstructed) ** 2)
    explained_variance_ratio = 1 - (unexplained_variance / total_variance)

    # Calculate topic coherence (using feature co-occurrence)
    coherence_scores = []
    for topic_idx in range(H.shape[0]):
        # Get top features for this topic
        top_features_idx = H[topic_idx].argsort()[::-1][:10]

        # Simple coherence measure: average pairwise correlation between top features
        coherence = 0
        count = 0
        for i in range(len(top_features_idx)):
            for j in range(i + 1, len(top_features_idx)):
                f1, f2 = top_features_idx[i], top_features_idx[j]
                # Calculate correlation between these features across documents
                corr = np.corrcoef(X_train[:, f1], X_train[:, f2])[0, 1]
                if not np.isnan(corr):
                    coherence += corr
                    count += 1

        coherence = coherence / count if count > 0 else 0
        coherence_scores.append(coherence)

    avg_coherence = np.mean(coherence_scores)
    print(f"  Train RMSE: {train_rmse:.4f}")
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Explained variance ratio: {explained_variance_ratio:.4f}")
    print(f"  Average topic coherence: {avg_coherence:.4f}")
