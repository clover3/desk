import json
import fire
import logging
import os
from collections import Counter

from tqdm import tqdm

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.infer_tokens import LOG, PatInferenceFirst
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def infer_tokens(sb="TwoXChromosomes"):
    print(sb)
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)

    t = 0.8
    t_strong = 0.93
    pos_ngram_list = []
    train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    n_item = 2000
    train_dataset = train_dataset.take(n_item)

    bert_tms = get_classifier("bert2_train_mix_sel")

    model_path = get_model_save_path(model_name)
    pat = PatInferenceFirst(model_path)
    for example in tqdm(train_dataset):
        text = example['text']
        tms_pred, tms_score = bert_tms(text)
        full_score = pat.get_full_text_score(text)
        domain_pred = int(full_score > 0.5)
        if full_score < t - 0.1:
            continue

        if int(tms_pred) == domain_pred:
            continue

        # print(f"{tms_pred} {domain_pred} {text}")
        # print("")
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
            pos_ngram_list.append(out_res)

    ngram_path2 = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_all", f"{sb}.json")
    json.dump(pos_ngram_list, open(ngram_path2, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(infer_tokens)
