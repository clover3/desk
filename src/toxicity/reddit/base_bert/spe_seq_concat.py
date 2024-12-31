import logging
import os
from functools import partial

import fire
from omegaconf import OmegaConf
from transformers import BertTokenizer, BertForSequenceClassification

from toxicity.io_helper import init_logging
from toxicity.reddit.base_bert.text_concat import preprocess_qd_concat
from toxicity.reddit.base_bert.train_clf_common import train_from_args
from toxicity.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from toxicity.reddit.train_common import get_default_training_argument, get_data_arguments
from toxicity.reddit.predict_split import predict_sb_split
from typing import Dict, Any
from transformers import PreTrainedTokenizer


LOG = logging.getLogger(__name__)



def spe_concat_exp(builder, debug, arch_class, run_name, dataset_name):
    # Given minimal information , prepare training args and models
    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)
    model = arch_class.from_pretrained(base_model)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokens = ["[unused{}]".format(i) for i in range(2000)]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    train_from_args(
        model,
        tokenizer,
        training_args,
        dataset_args,
        builder,
        run_name,
        dataset_name,
        preprocess_qd_concat,
        debug)


def get_sb_to_query():
    sb_to_idx = {}
    sb_to_q = {}
    def sb_to_query(sb: str) -> int:
        if sb not in sb_to_idx:
            idx = len(sb_to_idx)
            sb_to_idx[sb] = idx

            tokens = []
            st = idx * 10
            ed = (idx + 1) * 10
            for token_idx in range(st, ed):
                t = f"[unused{token_idx}]"
                tokens.append(t)

            q_new = " ".join(tokens)
            sb_to_q[sb] = q_new
        return sb_to_q[sb]
    return sb_to_query


def main(
        conf_path="",
        conf="",
        debug=False,
        do_sb_eval=False,
):
    init_logging()
    if conf:
        conf_path = conf
    conf = OmegaConf.load(conf_path)
    sb_to_query = get_sb_to_query()
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = BertForSequenceClassification
    spe_concat_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    if do_sb_eval:
        predict_sb_split(conf.run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
