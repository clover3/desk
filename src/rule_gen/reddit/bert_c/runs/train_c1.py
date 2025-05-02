import logging
import multiprocessing
from typing import Dict

import evaluate
import fire
import pandas as pd
from datasets import Dataset
from transformers import Trainer, TrainingArguments, BertTokenizer

from chair.misc_lib import rel
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.reddit_train_bert import build_training_argument, DataArguments
from rule_gen.reddit.bert_c.c_modeling import BertC1, C1Config
from rule_gen.reddit.bert_c.train_w_sb_ids import load_sb_name_to_id_mapping
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("TrainC")


def get_compute_metrics():
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        b_pred = logits > 0
        predictions = b_pred.astype(int)
        return clf_metrics.compute(predictions=predictions, references=labels)

    return compute_metrics


def load_triplet_dataset(file_path, mapping_dict: dict[str, int]):
    df = pd.read_csv(file_path,
                     na_filter=False, keep_default_na=False,
                     header=None, names=['sb_name', 'text', 'label'], dtype={"sb_name": str, "text": str, 'label': int})

    df['sb_id'] = df['sb_name'].map(mapping_dict)

    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            print(row)
            raise ValueError("Text field must contain string values")
    df = df.sample(frac=1).reset_index(drop=True)
    return Dataset.from_pandas(df)


def prepare_datasets(dataset_args: DataArguments, model_name):
    map_dict = load_sb_name_to_id_mapping()
    LOG.info(f"Loading training data from {rel(dataset_args.train_data_path)}")
    train_dataset = load_triplet_dataset(dataset_args.train_data_path, map_dict)
    LOG.info(f"Loading evaluation data from {rel(dataset_args.eval_data_path)}")
    eval_dataset = load_triplet_dataset(dataset_args.eval_data_path, map_dict)
    LOG.info("Creating datasets")
    if dataset_args.debug:
        LOG.info("Debug mode on. Take 100 items")
        train_dataset = train_dataset.take(100)
        eval_dataset = eval_dataset.take(100)

    # Load tokenizer and model
    LOG.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(
            examples['text'], padding='max_length',
            truncation=True, max_length=dataset_args.max_length)

    num_proc = multiprocessing.cpu_count()
    LOG.info(f"Using {num_proc} processes for dataset mapping")
    LOG.info("Tokenizing training dataset")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    LOG.info("Tokenizing evaluation dataset")
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    return tokenized_train, tokenized_eval


def train_c(
        model_name,
        model_factory,
        training_args: TrainingArguments,
        dataset_args,
        final_model_dir: str,
) -> Dict[str, float]:
    LOG.info("Starting training process")
    tokenized_train, tokenized_eval = prepare_datasets(dataset_args, model_name)
    model = model_factory()
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


def run_train(base_model, save_model_name, model_factory, debug):
    init_logging()
    sb = "train_comb4"
    data_name = "train_data2"
    with JobContext(save_model_name + "_train"):
        output_dir = get_model_save_path(save_model_name)
        final_model_dir = get_model_save_path(save_model_name)
        logging_dir = get_model_log_save_dir_path(save_model_name)
        max_length = 256
        training_args = build_training_argument(logging_dir, output_dir)
        training_args.num_train_epochs = 1
        dataset_args = DataArguments(
            train_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "train"),
            eval_data_path=get_reddit_train_data_path_ex(
                data_name, sb, "val"),
            max_length=max_length,
            debug=debug
        )

        eval_result = train_c(
            model_name=base_model,
            model_factory=model_factory,
            training_args=training_args,
            dataset_args=dataset_args,
            final_model_dir=final_model_dir,
        )

        proxy = get_task_manager_proxy()
        for metric in ["eval_f1"]:
            dataset = sb + "_val"
            metric_short = metric[len("eval_"):]
            proxy.report_number(save_model_name, eval_result[metric], dataset, metric_short)


def main(debug=False):
    base_model = 'bert-base-uncased'
    model_name = f"bert_c1"
    c_config = C1Config()
    model = BertC1(c_config)

    def model_factory():
        model = BertC1(c_config)
        return model

    run_train(base_model, model_name, model_factory, debug)


if __name__ == "__main__":
    fire.Fire(main)
