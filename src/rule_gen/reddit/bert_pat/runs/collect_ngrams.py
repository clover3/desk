import json

import fire
import logging
import os
from collections import Counter

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.infer_tokens import LOG, PatInferenceFirst
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.wrapper3 import JobContext


def infer_tokens(sb="TwoXChromosomes"):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)

    t = 0.8
    t_strong = 0.93
    n_item = 100
    confusion = Counter()
    pos_ngram_list = []
    neg_ngram_list = []
    train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    train_dataset = train_dataset.take(n_item)
    bert_tms = get_classifier("bert2_train_mix_sel")

    model_path = get_model_save_path(model_name)
    pat = PatInferenceFirst(model_path)
    for example in train_dataset:
        text = example['text']
        tms_pred, tms_score = bert_tms(text)
        full_score = pat.get_full_text_score(text)
        domain_pred = int(full_score > 0.5)
        confusion[(tms_pred, domain_pred)] += 1
        if full_score < t - 0.1:
            continue
        text_sp_rev = " ".join(text.split())
        pos_ngrams = []
        neg_ngrams = []
        for window_size in range(1, 4):
            ret = pat.get_first_view_scores(text, window_size=window_size)
            for result in ret:
                score = result["score"]
                out_res = {
                    'text': text_sp_rev,
                    'sub_text': result["sub_text"],
                    'score': score,
                }
                if score > t_strong:
                    pos_ngrams.append(out_res)
                elif score < 0.3:
                    neg_ngrams.append(out_res)

        if pos_ngrams:
            pos_ngram_list.append(pos_ngrams)
            neg_ngram_list.append(neg_ngrams)

    output = {
        "pos": pos_ngram_list,
        "neg": neg_ngram_list
    }
    save_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93", f"{sb}.json")
    make_parent_exists(save_path)
    json.dump(output, open(save_path, "w"))


if __name__ == "__main__":
    fire.Fire(infer_tokens)
