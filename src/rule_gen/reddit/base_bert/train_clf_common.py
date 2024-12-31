import logging

import evaluate
import numpy as np

from transformers import Trainer

from taskman_client.task_proxy import get_task_manager_proxy
from rule_gen.reddit.train_common import DatasetLoader, get_datasets_from_dataset_arg


LOG = logging.getLogger(__name__)



def get_compute_metrics():
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)
    return compute_metrics


def train_from_args(
        model,
        tokenizer,
        training_args,
        dataset_args,
        dataset_builder: DatasetLoader,
        run_name,
        dataset_name,
        preprocess_function,
        do_debug):
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    LOG.info("Training instance example: %s", str(train_dataset[0]))

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
    LOG.info(f"Saving fine-tuned model to {training_args.output_dir}")
    model.save_pretrained(training_args.output_dir)
    tokenizer.save_pretrained(training_args.output_dir)

    if not do_debug:
        metric = "eval_f1"
        proxy = get_task_manager_proxy()
        dataset = dataset_name + "_val"
        metric_short = metric[len("eval_"):]
        proxy.report_number(run_name, eval_results[metric], dataset, metric_short)
