import logging
import os
from torch import nn

from toxicity.reddit.proto.protory_net2 import ProtoryConfig
import fire
from omegaconf import OmegaConf
from transformers import PreTrainedTokenizer, Trainer

from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.io_helper import init_logging
from toxicity.reddit.train_common import ClfDatasetLoader, get_default_training_argument, get_data_arguments, \
    get_datasets_from_dataset_arg, DatasetLoader, train_classification_single_score, DataArguments
from toxicity.reddit.predict_split import predict_sb_split
from toxicity.reddit.proto.protory_net2 import ProtoryNet2
from typing import Dict, Any


LOG = logging.getLogger(__name__)


def preprocess_function(
        examples: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_length: int
) -> Dict[str, Any]:
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


def apply_tokenize(train_dataset, eval_dataset,
                   dataset_args: DataArguments,
                   tokenizer):

    def tokenize_name_inputs(examples):
        return preprocess_function(
            examples,
            tokenizer,
            dataset_args.max_length
        )

    tokenized_train = train_dataset.map(
        tokenize_name_inputs,
        batched=True,
        remove_columns=train_dataset.column_names,
        desc="Tokenizing training data"
    )

    tokenized_eval = None
    if eval_dataset is not None:
        tokenized_eval = eval_dataset.map(
            tokenize_name_inputs,
            batched=True,
            remove_columns=eval_dataset.column_names,
            desc="Tokenizing evaluation data"
        )

    return tokenized_train, tokenized_eval


class EncoderWrapper(nn.Module):
    def __init__(self, original_model):
        super().__init__()
        self.model = original_model

    def forward(self, input_ids, attention_mask):
        return self.model.encode_inputs(input_ids, attention_mask)


def train_bert_like_model(
        model,
        tokenizer,
        training_args,
        dataset_args,
        dataset_builder: DatasetLoader,
        run_name,
        dataset_name,
        do_debug):
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    LOG.info("Training instance example: %s", str(train_dataset[0]))
    tokenized_train, tokenized_eval = apply_tokenize(
        train_dataset, eval_dataset, dataset_args, tokenizer)

    LOG.info("init_prototypes")
    encoder = EncoderWrapper(model)
    trainer = Trainer(model=encoder,
                      args=training_args,
                      )
    outputs = trainer.predict(tokenized_train)
    sentence_embeddings_all = outputs.predictions
    LOG.info("Built embeddings: %s", str(sentence_embeddings_all.shape))
    model.init_prototypes(sentence_embeddings_all)
    eval_result = train_classification_single_score(
        model,
        training_args,
        tokenized_train,
        tokenized_eval
    )
    # if not do_debug:
    #     metric = "eval_f1"
    #     proxy = get_task_manager_proxy()
    #     dataset = dataset_name + "_val"
    #     metric_short = metric[len("eval_"):]
    #     proxy.report_number(run_name, eval_result[metric], dataset, metric_short)



def train_exp(dataset_builder, debug, conf):
    run_name = conf.run_name
    dataset_args = get_data_arguments(debug, conf.dataset_name)
    training_args = get_default_training_argument(run_name)
    training_args.learning_rate = conf.get('learning_rate', 1e-3)
    proton_config = ProtoryConfig(
        k_protos=conf.get('k_protos', 10),
        alpha=conf.get('alpha', 0.0001),
        beta=conf.get('beta', 0.01),
        lstm_dim=conf.get('lstm_dim', 128),
        base_model_name=conf.get('base_model_name', "sentence-transformers/all-MiniLM-L6-v2")
    )
    model = ProtoryNet2(proton_config)
    train_bert_like_model(
        model,
        model.tokenizer,
        training_args,
        dataset_args,
        dataset_builder,
        run_name,
        conf.dataset_name,
        debug)


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    conf_path = os.path.join("confs", "proto", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    LOG.info(str(conf))
    dataset_builder = ClfDatasetLoader()
    train_exp(
        dataset_builder,
        debug,
        conf,
    )

    # if do_sb_eval:
    #     predict_sb_split(run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
