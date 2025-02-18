import multiprocessing

import tqdm
import os
import json
import fire
from tqdm import tqdm
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv, init_logging
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.base_bert.train_clf_common import get_compute_metrics
from rule_gen.reddit.bert_pat.pat_modeling import BertPAT, CombineByScoreAdd
from rule_gen.reddit.bert_pat.scratch import tokenize_and_split
from rule_gen.reddit.bert_probe.probe_inference import ProbeInference
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
import logging
from typing import Dict

import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer

from rule_gen.reddit.bert_probe.probe_model import BertProbe
from rule_gen.reddit.base_bert.reddit_train_bert import prepare_datasets, build_training_argument, DataArguments
from chair.misc_lib import rel
from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("TrainPat")


def prepare_datasets(dataset_args: DataArguments, model_name):
    # Create datasets
    LOG.info(f"Loading training data from {rel(dataset_args.train_data_path)}")
    train_dataset = load_dataset_from_csv(dataset_args.train_data_path)
    LOG.info(f"Loading evaluation data from {rel(dataset_args.eval_data_path)}")
    eval_dataset = load_dataset_from_csv(dataset_args.eval_data_path)
    LOG.info("Creating datasets")
    # Load tokenizer and model
    LOG.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Tokenize function
    def process_example(example):
        result = tokenize_and_split(example['text'], tokenizer, dataset_args.max_length)
        result['label'] = example['label']
        return result

    # num_proc = multiprocessing.cpu_count()
    # LOG.info(f"Using {num_proc} processes for dataset mapping")
    # LOG.info("Tokenizing training dataset")
    num_proc = 1
    if dataset_args.debug:
        train_dataset = train_dataset.take(10)
    tokenized_train = train_dataset.map(process_example, batched=True, num_proc=num_proc)
    LOG.info("Tokenizing evaluation dataset")
    if dataset_args.debug:
        eval_dataset = eval_dataset.take(10)
    tokenized_eval = eval_dataset.map(process_example, batched=True, num_proc=num_proc)
    return tokenized_train, tokenized_eval


def train_two_seg(
        model_name,
        training_args: TrainingArguments,
        dataset_args,
        final_model_dir: str,
        num_labels: int = 2,
) -> Dict[str, float]:
    LOG.info("Starting training process")
    tokenized_train, tokenized_eval = prepare_datasets(dataset_args, model_name)
    model = BertPAT.from_pretrained(
        model_name,
        num_labels=num_labels,
        combine_layer_factory=CombineByScoreAdd,
    )
    # Initialize ProbeTrainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=get_compute_metrics(),
    )

    LOG.info("Starting training")
    train_result = trainer.train()
    LOG.info("Training completed")

    eval_results = trainer.evaluate(tokenized_eval)
    print("Evaluation results:", eval_results)

    trainer.save_model(training_args.output_dir)

    LOG.info(f"Training outputs and logs are saved in: {training_args.output_dir}")
    LOG.info(f"Final fine-tuned model and tokenizer are saved in: {final_model_dir}")
    LOG.info("BERT fine-tuning process completed")
    return eval_results


def reddit_train_pat_exp(sb = "TwoXChromosomes" , debug=False):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"

    with JobContext(model_name + "_train"):
        base_model = 'bert-base-uncased'
        output_dir = get_model_save_path(model_name)
        final_model_dir = get_model_save_path(model_name)
        logging_dir = get_model_log_save_dir_path(model_name)
        max_length = 256
        training_args = build_training_argument(logging_dir, output_dir)
        dataset_args = DataArguments(
            train_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "train"),
            eval_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "val"),
            max_length=max_length,
            debug=debug
        )
        eval_result = train_two_seg(
            model_name=base_model,
            training_args=training_args,
            dataset_args=dataset_args,
            final_model_dir=final_model_dir,
        )

        proxy = get_task_manager_proxy()
        for metric in ["eval_f1"]:
            dataset = sb + "_val"
            metric_short = metric[len("eval_"):]
            proxy.report_number(model_name, eval_result[metric], dataset, metric_short)


if __name__ == "__main__":
    fire.Fire(reddit_train_pat_exp)
