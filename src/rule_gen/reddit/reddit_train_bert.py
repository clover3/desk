from dataclasses import dataclass, field
from typing import Optional

import fire
import logging
from transformers import TrainingArguments, Trainer, BertTokenizer, BertForSequenceClassification
import multiprocessing

from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.train_clf_common import get_compute_metrics
from rule_gen.reddit.path_helper import get_reddit_train_data_path
from rule_gen.reddit.predict_clf import predict_clf_main
from rule_gen.reddit.train_bert import load_dataset_from_csv
from rule_gen.reddit.train_common import compute_per_device_batch_size

LOG = logging.getLogger(__name__)


# ... (keep all the existing import statements and helper functions)

@dataclass
class DataArguments:
    r"""
    Arguments pertaining to what data we are going to input our model for training and evaluation.
    """
    train_data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    max_length: int = field(default=256)


def finetune_bert(
        model_name,
        training_args,
        dataset_args,
        final_model_dir,
        num_labels=2,
):
    LOG.info("Starting BERT fine-tuning process")

    tokenized_train, tokenized_eval = prepare_datasets(dataset_args, model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Initialize Trainer
    LOG.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=get_compute_metrics(),
    )

    # Train the model
    LOG.info("Starting model training")
    train_result = trainer.train()
    LOG.info("Model training completed")
    eval_results = trainer.evaluate(tokenized_eval)
    print("eval_results", eval_results)

    # Save the model
    LOG.info(f"Saving fine-tuned model to {final_model_dir}")
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)

    LOG.info(f"Training outputs and logs are saved in: {training_args.output_dir}")
    LOG.info(f"Final fine-tuned model and tokenizer are saved in: {final_model_dir}")
    LOG.info("BERT fine-tuning process completed")
    return eval_results


def prepare_datasets(dataset_args: DataArguments, model_name):
    # Create datasets
    LOG.info(f"Loading training data from {dataset_args.train_data_path}")
    train_dataset = load_dataset_from_csv(dataset_args.train_data_path)
    LOG.info(f"Loading evaluation data from {dataset_args.eval_data_path}")
    eval_dataset = load_dataset_from_csv(dataset_args.eval_data_path)
    LOG.info("Creating datasets")
    # Load tokenizer and model
    LOG.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=dataset_args.max_length)

    # Determine number of processes to use
    num_proc = multiprocessing.cpu_count()
    LOG.info(f"Using {num_proc} processes for dataset mapping")
    # Tokenize datasets
    LOG.info("Tokenizing training dataset")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    LOG.info("Tokenizing evaluation dataset")
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    return tokenized_train, tokenized_eval


def build_training_argument(logging_dir, output_dir):
    # Set up training parameters
    num_train_epochs = 3
    learning_rate = 5e-5
    train_batch_size = 16
    eval_batch_size = 64
    warmup_ratio = 0.1
    # Load datasets to calculate total steps
    # Calculate total number of training steps
    per_device_batch_size = compute_per_device_batch_size(train_batch_size)
    LOG.info(f"Train/Per-device batch size: {train_batch_size}/{per_device_batch_size}")
    # Create training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=per_device_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_ratio=warmup_ratio,
        weight_decay=0.01,
        logging_dir=logging_dir,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )
    return training_args


def train_subreddit_classifier(subreddit="askscience_head"):
    init_logging()
    model_name = f"bert_{subreddit}"
    with JobContext(model_name + "_train"):
        base_model = 'bert-base-uncased'

        output_dir = get_model_save_path(model_name)
        final_model_dir = get_model_save_path(model_name)
        logging_dir = get_model_log_save_dir_path(model_name)
        max_length = 256
        training_args = build_training_argument(logging_dir, output_dir)
        dataset_args = DataArguments(
            train_data_path=get_reddit_train_data_path(subreddit, "train"),
            eval_data_path=get_reddit_train_data_path(subreddit, "val"),
            max_length=max_length
        )

        eval_result = finetune_bert(
            model_name=base_model,
            training_args=training_args,
            dataset_args=dataset_args,
            final_model_dir=final_model_dir,
        )

        proxy = get_task_manager_proxy()
        for metric in ["eval_f1"]:
            dataset = subreddit + "_val"
            metric_short = metric[len("eval_"):]
            proxy.report_number(model_name, eval_result[metric], dataset, metric_short)

        predict_clf_main(model_name, subreddit + "_val", do_eval=True, do_report=True)


# Example usage:
if __name__ == "__main__":
    fire.Fire(train_subreddit_classifier)
