from transformers import PreTrainedTokenizer, AutoTokenizer
from datasets import Dataset
import pandas as pd
from typing import List, Dict, Any

from transformers.tokenization_utils_base import PreTokenizedInput

from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.partition_util import random_token_split
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex




def tokenize_and_split(
        texts: list[str],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 512,
        return_tensor=None
) -> Dict[str, List | Any]:
    def partition(text) -> tuple[list[str], list[str]]:
        str_tokens = tokenizer.tokenize(text)
        first, second = random_token_split(str_tokens)
        return first, second

    first, second = zip(*map(partition, texts))
    first: List[PreTokenizedInput] = list(first)
    second: list[PreTokenizedInput] = list(second)

    tokenized1 = tokenizer(
        first,
        is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
        truncation=True,
        max_length=max_length,
        padding='max_length',
        return_tensors=return_tensor
    )
    tokenized2 = tokenizer(
        second,
        is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
        truncation=True,
        max_length=max_length,
        return_tensors=return_tensor,
        padding='max_length'
    )
    for i in range(len(tokenized1["input_ids"])):
        assert len(tokenized1['input_ids'][i]) <= max_length
    for i in range(len(tokenized2["input_ids"])):
        assert len(tokenized2['input_ids'][i]) <= max_length
    for i in range(len(tokenized1["attention_mask"])):
        assert len(tokenized1['attention_mask'][i]) <= max_length
    for i in range(len(tokenized2["input_ids"])):
        assert len(tokenized2['input_ids'][i]) <= max_length

    return {
        'input_ids1': tokenized1['input_ids'],
        'attention_mask1': tokenized1['attention_mask'],
        'input_ids2': tokenized2['input_ids'],
        'attention_mask2': tokenized2['attention_mask'],
    }


# Example usage:
if __name__ == "__main__":
    sb = "TwoXChromosomes"
    data_name = "train_data2"
    dataset_path = get_reddit_train_data_path_ex(data_name, sb, "train")
    model_name = "bert-base-uncased"  # or your preferred model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
