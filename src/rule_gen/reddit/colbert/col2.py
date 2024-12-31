import logging

import fire
from transformers import BertTokenizer

from desk_util.io_helper import init_logging
from rule_gen.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from rule_gen.reddit.colbert.modeling import Col2
from rule_gen.reddit.colbert.query_builders import get_sb_to_query
from rule_gen.reddit.colbert.train_common import train_bert_like_model
from rule_gen.reddit.train_common import get_default_training_argument, get_data_arguments

LOG = logging.getLogger(__name__)


def col_bert_train_exp(builder, debug, run_name, dataset_name):
    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)
    colbert = Col2.from_pretrained(base_model)
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


def main(debug=False, sb_strategy="name"):
    init_logging()
    sb_to_query = get_sb_to_query(sb_strategy)
    dataset_name = "train_comb1"
    builder = ThreeColumnDatasetLoader(sb_to_query)
    run_name = f"col2-{sb_strategy}"
    col_bert_train_exp(builder, debug, run_name, dataset_name)


if __name__ == "__main__":
    fire.Fire(main)
