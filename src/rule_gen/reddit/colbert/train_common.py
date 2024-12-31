import logging
from typing import Dict, Any

from transformers import PreTrainedTokenizer

from taskman_client.task_proxy import get_task_manager_proxy
from rule_gen.reddit.train_common import DataArguments, DatasetLoader, get_datasets_from_dataset_arg, \
    train_classification_single_score

LOG = logging.getLogger(__name__)


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


