import logging
import os

import fire
from omegaconf import OmegaConf
from transformers import BertTokenizer, BertForSequenceClassification

from desk_util.io_helper import init_logging
from rule_gen.reddit.base_bert.train_clf_common import train_from_args
from rule_gen.reddit.colbert.dataset_builder import ThreeColumnDatasetLoader
from rule_gen.reddit.colbert.query_builders import get_sb_to_query
from rule_gen.reddit.train_common import get_default_training_argument, get_data_arguments
from rule_gen.reddit.predict_split import predict_sb_split
from typing import Dict, Any
from transformers import PreTrainedTokenizer

LOG = logging.getLogger(__name__)


def preprocess_qd_concat(
        examples: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_length: int
) -> Dict[str, Any]:
    encodings = tokenizer(
        examples['query'],
        examples['document'],
        padding='max_length',  # Changed from 'padding=True'
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    # Convert labels to the correct format
    labels = examples['label']
    return {
        'input_ids': encodings['input_ids'],
        'attention_mask': encodings['attention_mask'],
        'labels': labels
    }


def text_concat_exp(builder, debug, arch_class, run_name, dataset_name):
    # Given minimal information , prepare training args and models

    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)
    model = arch_class.from_pretrained(base_model)
    tokenizer = BertTokenizer.from_pretrained(base_model)

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


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    conf_path = os.path.join("confs", "cross_encoder", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    print(conf)

    sb_to_query = get_sb_to_query(conf.sb_strategy)
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = BertForSequenceClassification
    text_concat_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    if do_sb_eval:
        predict_sb_split(run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
