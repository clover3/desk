import ast
import os
import pickle
import time
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from chair.misc_lib import group_by
from desk_util.io_helper import read_csv
# import seaborn as sns
# import matplotlib.pyplot as plt
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list


def analyze_pca_components(X, n_components=20):
    """
    Analyze the relationship between original features and PCA components using
    the components_ matrix directly from PCA.

    Args:
        X: Input data matrix of shape (n_samples, n_features)
        n_components: Number of PCA components to use

    Returns:
        dict: Dictionary containing PCA analysis results
    """
    # Create and fit the pipeline
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=n_components))
    ])

    # Fit the pipeline
    pipeline.fit(X)

    # Get the PCA object
    pca = pipeline.named_steps['pca']

    # Get components matrix (each row represents a principal component)
    # components_ contains the eigenvectors of the covariance matrix
    components = pca.components_

    X_transformed = pipeline.transform(X)

    # Create a DataFrame of components
    components_df = pd.DataFrame(
        components.T,  # Transpose to get features as rows
        columns=[f'PC{i + 1}' for i in range(n_components)]
    )


    # Find most influential features for each component
    top_features = {}
    for i in range(n_components):
        # Get absolute coefficients for this component
        abs_coeffs = np.abs(components[i])
        # Get indices of top 5 features
        top_indices = np.argsort(abs_coeffs)[-15:][::-1]
        top_features[f'PC{i + 1}'] = {
            'indices': top_indices,
            'coefficients': components[i, top_indices]
        }

    # Calculate explained variance ratio
    explained_variance_ratio = pca.explained_variance_ratio_

    # Get singular values (sqrt of eigenvalues)
    singular_values = pca.singular_values_

    return {
        'components': components_df,
        'top_features': top_features,
        'explained_variance_ratio': explained_variance_ratio,
        'singular_values': singular_values,
        'mean': pca.mean_,
        "X_transformed":X_transformed,
        'n_components': pca.n_components_
    }

def load_key():
    key_path = os.path.join(output_root_path, "reddit", "counter_train_data2_train_mix_key.csv")
    ret = read_csv(key_path)

    d = {}
    for text, counter_s in ret:
        d[text] = counter_s
        # ast.literal_eval(counter_s)
    return d
