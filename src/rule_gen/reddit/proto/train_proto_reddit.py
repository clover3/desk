import logging
import os
from torch import nn

from rule_gen.reddit.proto.protory_net2 import ProtoryConfig2, ProtoryNet3
import fire
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer, Trainer, TrainingArguments

from taskman_client.task_proxy import get_task_manager_proxy
from desk_util.io_helper import init_logging
from rule_gen.reddit.train_common import ClfDatasetLoader, get_default_training_argument, get_data_arguments, \
    get_datasets_from_dataset_arg, DatasetLoader, train_classification_single_score, DataArguments
from rule_gen.reddit.predict_split import predict_sb_split
from typing import Dict, Any


LOG = logging.getLogger(__name__)


def get_tokenize_formatter(
        tokenizer: PreTrainedTokenizer,
        max_length: int
):
    def tokenize_format(examples: Dict[str, Any]) -> Dict[str, Any]:
        encodings = tokenizer(
            examples['text'],
            padding='max_length',  # Changed from 'padding=True'
            truncation=True,
            max_length=max_length,
            return_tensors=None
        )
        labels = examples['label']
        return {
            'input_ids': encodings['input_ids'],
            'attention_mask': encodings['attention_mask'],
            'labels': labels
        }

    return tokenize_format


def apply_tokenize(train_dataset, eval_dataset, tokenize_format
                   ):
    tokenized_train = train_dataset.map(
        tokenize_format,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_format,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing evaluation data"
        )

    return tokenized_train, tokenized_eval


def initialize_init_prototype(model, tokenized_train, training_args):
    LOG.info("init_prototypes")
    class EncoderWrapper(nn.Module):
        def __init__(self, original_model):
            super().__init__()
            self.model = original_model

        def forward(self, input_ids, attention_mask):
            return self.model.encode_inputs(input_ids, attention_mask)

    encoder = EncoderWrapper(model)
    trainer = Trainer(model=encoder,
                      args=training_args,
                      )
    outputs = trainer.predict(tokenized_train)
    sentence_embeddings_all = outputs.predictions
    LOG.info("Built embeddings: %s", str(sentence_embeddings_all.shape))
    model.init_prototypes(sentence_embeddings_all)


def train_protory_net(
        model,
        tokenizer,
        training_args: TrainingArguments,
        dataset_args: DataArguments,
        dataset_builder: DatasetLoader,
        do_init: bool
):
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)

    LOG.info("Training instance example: %s", str(train_dataset[0]))
    tokenize_format = get_tokenize_formatter(tokenizer, dataset_args.max_length)

    tokenized_train, tokenized_eval = apply_tokenize(
        train_dataset, eval_dataset, tokenize_format)

    if do_init:
        initialize_init_prototype(model, tokenized_train, training_args)

    eval_result = train_classification_single_score(
        model,
        training_args,
        tokenized_train,
        tokenized_eval
    )
    return eval_result


def protonet_train_exp(model_cls, conf, do_sb_eval, debug):
    dataset_builder = ClfDatasetLoader()
    run_name = conf.run_name
    dataset_args = get_data_arguments(debug, conf.dataset_name)
    training_args = get_default_training_argument(run_name)
    training_args.learning_rate = conf.get('learning_rate', 1e-3)
    training_args.num_train_epochs = conf.get('epochs', 3)

    proton_config = ProtoryConfig2(
        k_protos=conf.get('k_protos', 10),
        alpha=conf.get('alpha', 0.0001),
        beta=conf.get('beta', 0.01),
        lstm_dim=conf.get('lstm_dim', 128),
        base_model_name=conf.get('base_model_name', "sentence-transformers/all-MiniLM-L6-v2")
    )
    do_init = conf.get('do_init', True)

    model = model_cls(proton_config)
    eval_result = train_protory_net(
        model,
        model.tokenizer,
        training_args,
        dataset_args,
        dataset_builder,
        do_init
    )

    if not debug:
        metric = "eval_f1"
        proxy = get_task_manager_proxy()
        dataset = conf.dataset_name + "_val"
        metric_short = metric[len("eval_"):]
        proxy.report_number(run_name, eval_result[metric], dataset, metric_short)

    if do_sb_eval:
        predict_sb_split(run_name + "/{}", "val")


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    conf_path = os.path.join("confs", "proto", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    LOG.info(str(conf))
    protonet_train_exp(ProtoryNet3, conf, do_sb_eval, debug)


if __name__ == "__main__":
    fire.Fire(main)
