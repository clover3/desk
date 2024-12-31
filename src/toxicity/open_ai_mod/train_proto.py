import numpy as np
import logging

import evaluate
import fire
from omegaconf import OmegaConf
from transformers import TrainingArguments, BertPreTrainedModel, Trainer

from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.io_helper import init_logging
from toxicity.open_ai_mod.path_helper import get_open_ai_mod_csv_path
from toxicity.reddit.proto.protory_net2 import ProtoryConfig2, ProtoryNet2
from toxicity.reddit.proto.train_proto_reddit import initialize_init_prototype, \
    get_tokenize_formatter, apply_tokenize
from toxicity.reddit.train_common import ClfDatasetLoader, get_default_training_argument, \
    DataArguments, get_datasets_from_dataset_arg, DatasetLoader

LOG = logging.getLogger(__name__)


def get_data_arguments(do_debug):
    if do_debug:
        n_train_sample = 100
        n_eval_sample = 10
    else:
        n_train_sample = None
        n_eval_sample = None

    dataset_args = DataArguments(
        train_data_path=get_open_ai_mod_csv_path("train"),
        eval_data_path=get_open_ai_mod_csv_path("val"),
        max_length=512,
        n_train_sample=n_train_sample,
        n_eval_sample=n_eval_sample,
    )
    return dataset_args


def train_classification_single_score(
        model: BertPreTrainedModel,
        training_args: TrainingArguments,
        tokenized_train,
        tokenized_eval,
):
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred

        def sigmoid(z):
            return 1 / (1 + np.exp(-z))
        probs = sigmoid(logits)
        predictions = probs > 0.5
        return clf_metrics.compute(predictions=predictions, references=labels)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
    )
    LOG.info("Starting training...")
    trainer.train()
    eval_results = trainer.evaluate(tokenized_eval)
    print("eval_results", eval_results)

    trainer.save_model(training_args.output_dir)
    LOG.info("Training completed")

    return eval_results


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


def protonet_train_exp(model_cls, conf, debug):
    dataset_builder = ClfDatasetLoader()
    run_name = conf.run_name
    dataset_args = get_data_arguments(debug)
    training_args = get_default_training_argument(run_name)
    training_args.learning_rate = conf.get('learning_rate', 1e-3)
    training_args.num_train_epochs = conf.get('epochs', 10)

    proton_config = ProtoryConfig2(
        k_protos=conf.get('k_protos', 10),
        alpha=conf.get('alpha', 0.0001),
        beta=conf.get('beta', 0.01),
        lstm_dim=conf.get('lstm_dim', 128),
        base_model_name=conf.get('base_model_name', "sentence-transformers/all-MiniLM-L6-v2")
    )
    LOG.info("proton_config=%s", str(proton_config))
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
        return eval_result[metric]


def main(
        conf_path="",
        debug=False,
):
    init_logging()
    conf = OmegaConf.load(conf_path)
    LOG.info(str(conf))
    protonet_train_exp(ProtoryNet2, conf, debug)


if __name__ == "__main__":
    fire.Fire(main)
