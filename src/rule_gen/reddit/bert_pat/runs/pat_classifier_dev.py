import logging

import fire
import tqdm
from tqdm import tqdm

from chair.misc_lib import SuccessCounter
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.infer_tokens import LOG
from rule_gen.reddit.bert_pat.pat_classifier import PatBasedClassifier
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def infer_tokens(sb="TwoXChromosomes"):
    print(sb)
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)

    train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    n_item = 100
    train_dataset = train_dataset.take(n_item)
    model_path = get_model_save_path(model_name)
    pat = PatBasedClassifier(model_path)
    sc = SuccessCounter()
    for example in tqdm(train_dataset):
        text = example['text']
        p1 = pat.classify(text, "logit_avg", [3])
        p2 = pat.get_full_text_score(text)
        pred1 = p1 > 0.5
        pred2 = p2 > 0.5
        sc.add(pred1 == pred2)
    print(sc.get_summary())


if __name__ == "__main__":
    fire.Fire(infer_tokens)
