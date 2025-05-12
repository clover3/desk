import tqdm
import json
import pickle
from typing import Callable

from chair.misc_lib import make_parent_exists
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
import os
from sklearn.linear_model import LogisticRegression

from desk_util.io_helper import read_csv
from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst
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

def main(sb="TwoXChromosomes"):
    get_feature = get_s9_combined()
    data_name = "train_data2"
    n_item = 2000
    skip = 100
    data = read_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    data = data[skip:skip + n_item]
    model_path = os.path.join(output_root_path, "models", "sklearn_run4", f"{sb}.pickle")
    clf = pickle.load(open(model_path, "rb"))
    t_strong = 0.9

    data_itr = tqdm.tqdm(data)
    model_name = f"bert_ts_{sb}"
    pat = PatInferenceFirst(get_model_save_path(model_name))
    itr = enum_false_negative(data_itr, clf, get_feature)
    output = []
    for text in itr:
        full_score = pat.get_full_text_score(text)
        domain_pred = int(full_score > 0.5)
        if not domain_pred:
            continue
        text_sp_rev = " ".join(text.split())
        strong_sub_texts = []
        for window_size in [1, 3, 5]:
            ret = pat.get_first_view_scores(text, window_size=window_size)
            for result in ret:
                score = result["score"]
                if score > t_strong:
                    sub_text = result["sub_text"]
                    strong_sub_texts.append({
                        'sub_text': sub_text,
                        'score': score,
                    })
        if strong_sub_texts:
            out_res = {
                'text': text_sp_rev,
                'strong_sub_texts': strong_sub_texts,
            }
            output.append(out_res)
        # print({"text": text})
    save_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93", f"{sb}.json")
    make_parent_exists(save_path)
    json.dump(output, open(save_path, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
