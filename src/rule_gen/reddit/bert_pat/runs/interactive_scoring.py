import numpy as np
import logging

import fire

from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_pat.infer_tokens import LOG
from rule_gen.reddit.bert_pat.pat_classifier import PatBasedClassifier
from scipy.special import softmax


def infer_tokens(sb="TwoXChromosomes"):
    print(sb)
    init_logging()
    model_name = f"bert_ts_{sb}"
    print(model_name)
    LOG.setLevel(logging.INFO)
    model_path = get_model_save_path(model_name)
    pat = PatBasedClassifier(model_path)
    while True:
        text = input("Please enter your text: ")
        items = pat.get_sub_text_scores(text)
        s_list = []
        for sub_text, logit in items:
            probs = softmax(logit[0])
            score = probs[1]
            s = "{}: {:.2f}".format(sub_text, score)
            s_list.append(s)
        print(" ".join(s_list))


if __name__ == "__main__":
    fire.Fire(infer_tokens)
