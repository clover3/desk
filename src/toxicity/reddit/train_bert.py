import fire
import numpy as np

from toxicity.io_helper import init_logging
from toxicity.path_helper import get_model_save_path, get_reddit_train_data_path, get_model_log_save_dir_path

import pandas as pd
import math
import logging
import multiprocessing
from transformers import BertTokenizer, BertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import evaluate

from toxicity.reddit.train_common import compute_per_device_batch_size

LOG = logging.getLogger(__name__)


def load_dataset_from_csv(data_path):
    df = pd.read_csv(data_path,
                     na_filter=False, keep_default_na=False,
                     header=None, names=['text', 'label'], dtype={"text": str, 'label': int})
    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            print(row)
            raise ValueError
    return Dataset.from_pandas(df)


def finetune_bert(
        train_data_path,
        eval_data_path,
        model_name='bert-base-uncased',
        output_dir='./training_outputs',
        final_model_dir='./fine_tuned_bert',
        logging_dir='./logs',
        num_train_epochs=3,
        learning_rate=5e-5,
        train_batch_size=16,
        eval_batch_size=64,
        max_length=128,
        warmup_ratio=0.1,
        num_labels=2,
        num_proc=None  # New parameter for controlling number of processes
):
    LOG.info("Starting BERT fine-tuning process")
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    # Create datasets
    LOG.info(f"Loading training data from {train_data_path}")
    train_dataset = load_dataset_from_csv(train_data_path)
    LOG.info(f"Loading evaluation data from {eval_data_path}")

    eval_dataset = load_dataset_from_csv(eval_data_path)
    LOG.info("Creating datasets")

    # Load tokenizer and model
    LOG.info(f"Loading tokenizer and model: {model_name}")
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)

    # Tokenize function
    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    # Determine number of processes to use
    if num_proc is None:
        num_proc = multiprocessing.cpu_count()  # Use all available CPUs
    LOG.info(f"Using {num_proc} processes for dataset mapping")

    # Tokenize datasets
    LOG.info("Tokenizing training dataset")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, num_proc=num_proc)
    LOG.info("Tokenizing evaluation dataset")
    tokenized_eval = eval_dataset.map(tokenize_function, batched=True, num_proc=num_proc)

    # Calculate total number of training steps
    total_train_steps = math.ceil(len(tokenized_train) / train_batch_size) * num_train_epochs
    per_device_batch_size = compute_per_device_batch_size(train_batch_size)
    LOG.info(f"Train/Per-device batch size: {train_batch_size}/{per_device_batch_size}")

    warmup_steps = math.ceil(total_train_steps * warmup_ratio)
    LOG.info(f"Total training steps: {total_train_steps}")
    LOG.info(f"Warmup steps: {warmup_steps}")

    # Training arguments
    LOG.info("Setting up training arguments")
    training_args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=num_train_epochs,
        per_device_train_batch_size=train_batch_size,
        per_device_eval_batch_size=eval_batch_size,
        learning_rate=learning_rate,
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        logging_dir=logging_dir,
        logging_steps=10,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
    )

    # Initialize Trainer
    LOG.info("Initializing Trainer")
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_eval,
        compute_metrics=compute_metrics,
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

    LOG.info(f"Training outputs and logs are saved in: {output_dir}")
    LOG.info(f"Final fine-tuned model and tokenizer are saved in: {final_model_dir}")
    LOG.info("BERT fine-tuning process completed")
    return eval_results


def train_subreddit_classifier(subreddit="The_Donald"):
    init_logging()
    model_name = f"bert_{subreddit}"
    return finetune_bert(
        train_data_path=get_reddit_train_data_path(subreddit, "train"),
        eval_data_path=get_reddit_train_data_path(subreddit, "val"),
        model_name='bert-base-uncased',
        output_dir=get_model_save_path(model_name),
        final_model_dir=get_model_save_path(model_name),
        logging_dir=get_model_log_save_dir_path(model_name),
        num_train_epochs=3,
        learning_rate=5e-5,
        train_batch_size=16,
        eval_batch_size=64,
        max_length=256,
        warmup_ratio=0.1,  # 10% of total steps for warmup,
    )


# Example usage:
if __name__ == "__main__":
    fire.Fire(train_subreddit_classifier)
