import logging
import os

import fire
import yaml
from omegaconf import OmegaConf
from transformers import BertTokenizer

from toxicity.io_helper import init_logging
from toxicity.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from toxicity.reddit.colbert.modeling import Col3, ColBertForSequenceClassification
from toxicity.reddit.colbert.query_builders import get_sb_to_query
from toxicity.reddit.colbert.train_common import get_default_training_argument, \
    train_bert_like_model, get_data_arguments

LOG = logging.getLogger(__name__)


def col_bert_train_exp(builder, debug, arch_class, run_name, dataset_name):
    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)
    colbert = arch_class.from_pretrained(base_model)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    colbert.colbert_set_up(tokenizer)

    train_bert_like_model(
        colbert,
        tokenizer,
        training_args,
        dataset_args,
        builder,
        run_name,
        dataset_name,
        debug)


def get_arch_class(arch_name):
    return {
        "col1": ColBertForSequenceClassification,
        "col3": Col3
    }[arch_name]


def main(
        debug=False,
        run_name=""):
    init_logging()
    conf_path = os.path.join("confs", "col", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    print(conf)

    sb_to_query = get_sb_to_query(conf.sb_strategy)
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = get_arch_class(conf.arch_name)
    col_bert_train_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)


if __name__ == "__main__":
    fire.Fire(main)