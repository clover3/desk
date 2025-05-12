import tqdm
import pickle
from typing import Callable

from rule_gen.cpath import output_root_path
import os
from sklearn.linear_model import LogisticRegression

from desk_util.io_helper import read_csv
from rule_gen.reddit.s9.s9_loader import get_s9_combined
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def enum_false_negative(text_label_itr, clf: LogisticRegression, feature_fn: Callable[[str], list[int]]):
    for text, label in text_label_itr:
        if not label:
            continue

        feature = feature_fn(text)
        pred = clf.predict([feature])
        if not pred:
            yield text

import fire

def main(sb= "TwoXChromosomes"):
    get_feature = get_s9_combined()
    data_name = "train_data2"
    n_item = 100
    skip = 100
    data = read_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    data = data[skip:skip + n_item]
    model_path = os.path.join(output_root_path, "models", "sklearn_run4", f"{sb}.pickle")
    clf = pickle.load(open(model_path, "rb"))
    data_itr = tqdm.tqdm(data)
    model_name = f"bert_ts_{sb}"
    # pat = PatInferenceFirst(get_model_save_path(model_name))
    itr = enum_false_negative(data_itr, clf, get_feature)
    for text in itr:
        print({"text": text})


if __name__ == "__main__":
    fire.Fire(main)