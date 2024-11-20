import logging
import os

import fire
from omegaconf import OmegaConf
from transformers import BertTokenizer

from toxicity.io_helper import init_logging
from toxicity.reddit.colbert.compute_var import compute_stdev
from toxicity.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from toxicity.reddit.colbert.modeling import get_arch_class
from toxicity.reddit.colbert.query_builders import get_sb_to_query
from toxicity.reddit.colbert.train_common import get_default_training_argument, \
    train_bert_like_model, get_data_arguments
from toxicity.reddit.predict_split import predict_sb_split

LOG = logging.getLogger(__name__)


def col_bert_train_exp(builder, debug, arch_class, run_name, dataset_name):
    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)

    train_bert_like_model(
        colbert,
        tokenizer,
        training_args,
        dataset_args,
        builder,
        run_name,
        dataset_name,
        debug)


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    col_bert_train_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    compute_stdev(run_name + "/hearthstone", None)
    if do_sb_eval:
        predict_sb_split(run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
