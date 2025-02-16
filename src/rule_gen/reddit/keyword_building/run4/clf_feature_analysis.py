from chair.tab_print import print_table
from rule_gen.cpath import output_root_path
import os
import pickle

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
import time


def analyze_variance_ratios(explained_variance_ratio):
    """
    Analyze explained variance ratios for different thresholds
    """
    cumulative_variance = np.cumsum(explained_variance_ratio)
    thresholds = [0.8, 0.9, 0.95, 0.98, 0.99]
    components_needed = {}

    for threshold in thresholds:
        n_components = np.argmax(cumulative_variance >= threshold) + 1
        components_needed[threshold] = n_components

    return components_needed


def measure_pca_performance(X, n_components=None):
    """
    Measure PCA performance including time and reconstruction error
    """
    # Create pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])

    # Measure fit transform time
    start_time = time.time()
    X_transformed = pipeline.fit_transform(X)
    print("X_transformed", X_transformed.shape)
    fit_transform_time = time.time() - start_time

    # Measure reconstruction time and error
    start_time = time.time()
    X_reconstructed = pipeline.inverse_transform(X_transformed)
    reconstruction_time = time.time() - start_time

    # Calculate reconstruction error
    reconstruction_error = np.mean(np.square(X - X_reconstructed))

    # Get PCA object
    pca = pipeline.named_steps['pca']

    # Calculate metrics
    performance_metrics = {
        'fit_transform_time': fit_transform_time,
        'reconstruction_time': reconstruction_time,
        'reconstruction_error': reconstruction_error,
        'n_components': pca.n_components_,
        'explained_variance_ratio': pca.explained_variance_ratio_.sum(),
        'input_shape': X.shape,
        'output_shape': X_transformed.shape,
        'compression_ratio': X_transformed.size / X.size
    }

    return performance_metrics, pca, X_transformed


def main():
    # Generate random data
    print("Loading data...")
    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    X = pickle.load(open(feature_save_path, "rb"))
    print(X.shape)

    # Run PCA with different numbers of components
    n_components_list = [5, 10, 15, 20, 30, 50]
    results = {}

    print("\nTesting different numbers of components:")
    cols = ["n_comp",
            "reconstruction_error",
            "explained_variance_ratio",
            "compression_ratio"
            ]
    table = [cols]
    for n_comp in n_components_list:
        metrics, pca, X_transformed = measure_pca_performance(X, n_components=n_comp)
        results[n_comp] = metrics
        row = [n_comp]

        for key in cols[1:]:
            row.append(("{0:.3f}".format(metrics[key])))
        table.append(row)
    print_table(table)
    # Full PCA analysis

    print("\nAnalyzing optimal number of components...")
    metrics, pca, _ = measure_pca_performance(X)
    components_needed = analyze_variance_ratios(pca.explained_variance_ratio_)

    print("\nOptimal number of components for different variance thresholds:")
    for threshold, n_components in components_needed.items():
        print(f"For {threshold * 100}% variance explained: {n_components}")


if __name__ == "__main__":
    main()
