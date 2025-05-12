import pickle

import tqdm

from chair.misc_lib import make_parent_exists
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
import os


import fire

from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        model_name = f"bert_ts_{sb}"
        pat = PatInferenceFirst(get_model_save_path(model_name))
        for i in [1, 2]:
            voca_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                     "s9_ngram_93_voca", f"{i}.txt")
            if not os.path.exists(voca_path):
                print(voca_path, "does not exist")
                continue
            save_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca_score", f"{sb}.{i}.pkl")
            if os.path.exists(save_path):
                print(save_path, "already not exist")
                continue

            make_parent_exists(save_path)
            voca = [l.strip() for l in open(voca_path, "r")]
            data_itr = tqdm.tqdm(voca)
            scores = []
            for t in data_itr:
                scores.append(pat.get_full_text_score(t))
            pickle.dump(scores, open(save_path, "wb"))


if __name__ == "__main__":
    main()