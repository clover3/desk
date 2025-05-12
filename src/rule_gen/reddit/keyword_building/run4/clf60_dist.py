import os
import pickle
from collections import Counter

import numpy as np

from rule_gen.cpath import output_root_path


def main():
    # Load data
    print("Loading data...")
    feature_save_path = os.path.join(output_root_path, "reddit", "pickles", "60clf.pkl")
    X = pickle.load(open(feature_save_path, "rb"))
    T = np.sum(X, axis=1)
    counter = Counter()
    for v in T:
        counter[v] += 1

    for i in range(X.shape[1]):
        print(i, counter[i])


if __name__ == "__main__":
    main()

