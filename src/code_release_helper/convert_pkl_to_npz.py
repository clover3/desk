import os
import pickle

import numpy as np
import tqdm

from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def convert_pkl_to_npz(n=1):
    """Convert pickle files containing score lists to compressed npz format"""
    subreddit_list = get_split_subreddit_list("train")

    for sb in tqdm.tqdm(subreddit_list, desc="Converting subreddits"):
        pkl_path = get_rp_path("sb_term_scores", f"{sb}.{n}.pkl")
        npz_path = get_rp_path("sb_term_scores", f"{sb}.{n}.npz")

        if not os.path.exists(pkl_path):
            print(f"Skipping {pkl_path} - does not exist")
            continue

        if os.path.exists(npz_path):
            print(f"Skipping {npz_path} - already exists")
            continue

        # Load pickle
        with open(pkl_path, 'rb') as f:
            scores = pickle.load(f)

        # Convert to numpy array and save as compressed npz
        scores_array = np.array(scores, dtype=np.float32)
        np.savez_compressed(npz_path, scores=scores_array)

        print(f"Converted {pkl_path} -> {npz_path}")
        print(f"  Shape: {scores_array.shape}, Dtype: {scores_array.dtype}")


if __name__ == "__main__":
    for n in range(1, 10):
        convert_pkl_to_npz(n)