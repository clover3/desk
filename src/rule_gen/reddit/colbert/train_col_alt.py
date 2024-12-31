import logging
import os

import fire
from omegaconf import OmegaConf
from transformers import BertTokenizer

from desk_util.io_helper import init_logging
from rule_gen.reddit.colbert.compute_var import compute_stdev
from rule_gen.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from rule_gen.reddit.colbert.modeling import get_arch_class
from rule_gen.reddit.colbert.query_builders import get_sb_to_query
from rule_gen.reddit.colbert.train_common import train_bert_like_model
from rule_gen.reddit.train_common import get_default_training_argument, get_data_arguments
from rule_gen.reddit.predict_split import predict_sb_split

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


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    conf_path = os.path.join("confs", "col", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    print(conf)

    sb_to_query = get_sb_to_query(conf.sb_strategy)
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = get_arch_class(conf.arch_name)
    col_bert_train_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    compute_stdev(run_name + "/hearthstone", None)
    if do_sb_eval:
        predict_sb_split(run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
