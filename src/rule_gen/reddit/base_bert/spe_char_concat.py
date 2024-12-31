import logging
from functools import partial

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


def preprocess_spe_concat(
        query_to_id,
        examples: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_length: int
) -> Dict[str, Any]:
    query_new = []
    for q in examples['query']:
        idx: int = query_to_id(q)
        q_new = f"[unused{idx}]"

        assert q_new in tokenizer.get_vocab()
        query_new.append(q_new)

    encodings = tokenizer(
        query_new,
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


class QtoIdx:
    """
    A class that maps query strings to unique integer indices.
    New queries are assigned incremental indices starting from 0.
    """

    def __init__(self):
        """
        Initialize an empty query-to-index mapping dictionary.
        """
        self.q_to_idx = {}
        # Add reverse mapping to get query from index
        self.idx_to_q = {}

    def query_to_id(self, query: str) -> int:
        if query not in self.q_to_idx:
            idx = len(self.q_to_idx)
            self.q_to_idx[query] = idx
            self.idx_to_q[idx] = query

        return self.q_to_idx[query]

    def id_to_query(self, idx: int) -> str:
        if idx not in self.idx_to_q:
            raise KeyError(f"Index {idx} not found in mapping")
        return self.idx_to_q[idx]

    def __len__(self) -> int:
        return len(self.q_to_idx)

def spe_concat_exp(builder, debug, arch_class, run_name, dataset_name):
    # Given minimal information , prepare training args and models
    base_model = 'bert-base-uncased'
    dataset_args = get_data_arguments(debug, dataset_name)
    training_args = get_default_training_argument(run_name)
    model = arch_class.from_pretrained(base_model)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokens = ["[unused{}]".format(i) for i in range(200)]
    tokenizer.add_special_tokens({"additional_special_tokens": tokens})

    q_to_idx = QtoIdx()
    preprocess_fn = partial(preprocess_spe_concat,
                            q_to_idx.query_to_id)
    train_from_args(
        model,
        tokenizer,
        training_args,
        dataset_args,
        builder,
        run_name,
        dataset_name,
        preprocess_fn,
        debug)


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
    sb_to_query = get_sb_to_query(conf.sb_strategy)
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = BertForSequenceClassification
    spe_concat_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    if do_sb_eval:
        predict_sb_split(conf.run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
