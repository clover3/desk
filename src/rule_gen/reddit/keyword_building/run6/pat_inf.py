import os

import fire
import numpy as np
import tqdm

from chair.misc_lib import make_parent_exists
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main(n=1):
    term_list = load_run6_10k_terms(n)
    subreddit_list = get_split_subreddit_list("train")

    for sb in subreddit_list:
        model_name = f"bert_ts_{sb}"
        pat = PatInferenceFirst(get_model_save_path(model_name))
        save_path = get_rp_path("sb_term_scores", f"{sb}.{n}.npz")  # Changed extension

        if os.path.exists(save_path):
            print(save_path, "already exist")
            continue

        make_parent_exists(save_path)
        data_itr = tqdm.tqdm(term_list)
        scores = []
        for t in data_itr:
            text = t if n == 1 else " ".join(t)
            scores.append(pat.get_full_text_score(text))

        # Save as compressed npz instead of pickle
        scores_array = np.array(scores, dtype=np.float32)
        np.savez_compressed(save_path, scores=scores_array)


if __name__ == "__main__":
    fire.Fire(main)
