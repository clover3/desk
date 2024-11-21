import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

import evaluate
import pandas as pd
import torch
from datasets import Dataset as HFDataset
from transformers import TrainingArguments, BertPreTrainedModel, Trainer

from toxicity.path_helper import get_model_save_path, get_model_log_save_dir_path
from toxicity.reddit.path_helper import get_reddit_train_data_path

LOG = logging.getLogger(__name__)

def compute_per_device_batch_size(train_batch_size):
    # Get the number of available GPUs
    num_gpus = torch.cuda.device_count()

    # If no GPUs are available, set num_gpus to 1 (for CPU)
    if num_gpus == 0:
        num_gpus = 1

    # Compute per_device_train_batch_size
    per_device_train_batch_size = train_batch_size // num_gpus

    # Ensure per_device_train_batch_size is at least 1
    per_device_train_batch_size = max(1, per_device_train_batch_size)

    return per_device_train_batch_size

@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    max_length: int = field(default=256)
    n_train_sample: int = field(default=10000000)
    n_eval_sample: int = field(default=1000)


class DatasetLoader(ABC):
    @abstractmethod
    def get(self, data_path, max_samples=None) -> HFDataset:
        pass


def get_datasets_from_dataset_arg(
        builder: DatasetLoader, dataset_args: DataArguments):
    LOG.info(f"Loading training data from {dataset_args.train_data_path}")
    train_dataset = builder.get(
        dataset_args.train_data_path,
        dataset_args.n_train_sample,
    )
    LOG.info(f"Loading evaluation data from {dataset_args.eval_data_path}")
    eval_dataset = builder.get(
        dataset_args.eval_data_path,
        dataset_args.n_eval_sample
    )
    return train_dataset, eval_dataset


class ClfDatasetLoader(DatasetLoader):
    def get(self, data_path, max_samples=None):
        df = pd.read_csv(
            data_path,
            na_filter=False,
            keep_default_na=False,
            header=None,
            names=['text', 'label'],
            dtype={"text": str, 'label': int}
        )
        print(df.head(10))
        if max_samples is not None:
            df = df.head(max_samples)
        return HFDataset.from_pandas(df)


def get_reddit_data_arguments(do_debug, subreddit):
    if do_debug:
        n_train_sample = 100
        n_eval_sample = 10
    else:
        n_train_sample = 10000
        n_eval_sample = 1000
    dataset_args = DataArguments(
        train_data_path=get_reddit_train_data_path(subreddit, "train"),
        eval_data_path=get_reddit_train_data_path(subreddit, "val"),
        max_length=512,
        n_train_sample=n_train_sample,
        n_eval_sample=n_eval_sample,
    )
    return dataset_args


def get_default_training_argument(model_name):
    training_args = TrainingArguments(
        output_dir=get_model_save_path(model_name),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=5e-5,
        warmup_ratio=0.1,
        weight_decay=0.01,
        logging_dir=get_model_log_save_dir_path(model_name),
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1"
    )
    return training_args


def get_data_arguments(do_debug, name):
    if do_debug:
        n_train_sample = 100
        n_eval_sample = 10
    else:
        n_train_sample = None
        n_eval_sample = None

    dataset_args = DataArguments(
        train_data_path=get_reddit_train_data_path(name, "train"),
        eval_data_path=get_reddit_train_data_path(name, "val"),
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
        predictions = logits > 0.5
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
