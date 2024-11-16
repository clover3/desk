import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

import evaluate
from transformers import PreTrainedTokenizer, BertTokenizer, BertPreTrainedModel, TrainingArguments, Trainer

from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.path_helper import get_model_save_path, get_model_log_save_dir_path
from toxicity.reddit.colbert.dataset_builder import TwoColumnDatasetLoader
from toxicity.reddit.path_helper import get_reddit_train_data_path

LOG = logging.getLogger(__name__)


@dataclass
class DataArguments:
    train_data_path: str = field(default=None)
    eval_data_path: Optional[str] = field(default=None)
    max_length: int = field(default=256)
    n_train_sample: int = field(default=10000000)
    n_eval_sample: int = field(default=1000)


def preprocess_function(
        examples: Dict[str, Any],
        tokenizer: PreTrainedTokenizer,
        max_length: int
) -> Dict[str, Any]:
    query_encodings = tokenizer(
        examples['query'],
        padding='max_length',  # Changed from 'padding=True'
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    doc_encodings = tokenizer(
        examples['document'],
        padding='max_length',  # Changed from 'padding=True'
        truncation=True,
        max_length=max_length,
        return_tensors=None
    )

    # Convert labels to the correct format
    labels = examples['label']

    return {
        'query_input_ids': query_encodings['input_ids'],
        'query_attention_mask': query_encodings['attention_mask'],
        'doc_input_ids': doc_encodings['input_ids'],
        'doc_attention_mask': doc_encodings['attention_mask'],
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


def get_datasets_from_dataset_arg(builder: TwoColumnDatasetLoader, dataset_args: DataArguments):
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


def train_bert_like_model(
        model, tokenizer,
        training_args,
        dataset_args,
        dataset_builder,
        run_name,
        dataset_name,
        do_debug):
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    LOG.info("Training instance example: %s", str(train_dataset[0]))
    tokenized_train, tokenized_eval = apply_tokenize(
        train_dataset, eval_dataset, dataset_args, tokenizer)
    eval_result = train_classification_single_score(
        model,
        training_args,
        tokenized_train,
        tokenized_eval
    )
    if not do_debug:
        metric = "eval_f1"
        proxy = get_task_manager_proxy()
        dataset = dataset_name + "_val"
        metric_short = metric[len("eval_"):]
        proxy.report_number(run_name, eval_result[metric], dataset, metric_short)


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
