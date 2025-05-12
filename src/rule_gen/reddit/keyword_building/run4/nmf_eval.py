import os
import pickle
import numpy as np
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.run4.nmf_eval_common import run_nmf_eval
from rule_gen.reddit.keyword_building.run4.pca import load_key
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list
from sklearn.decomposition import NMF, PCA
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split



def main():
    # Load data
    print("Loading data...")
    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    X = pickle.load(open(feature_save_path, "rb"))
    print(f"Data shape: {X.shape}")
    n_comp = 1
    model = PCA(n_components=n_comp)
    X = X - np.mean(X, axis=1, keepdims=True)
    run_nmf_eval(model, X, n_comp)


if __name__ == "__main__":
    main()