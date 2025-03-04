

import time
from collections import Counter

import torch
import logging

import fire
from transformers import AutoTokenizer

from chair.misc_lib import Averager, print_dict_tab
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.partition_util import get_non_sharp_indices, random_token_split
from rule_gen.reddit.bert_pat.pat_modeling import BertPAT, CombineByScoreAdd, BertPatFirst
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("InfPat")


class PatInference:
    def __init__(self, model_path, debug=False, window_size=1):
        self.debug = debug
        self.model_path =  model_path
        self.max_length = 256
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_model()
        self.window_size = window_size

    def _init_model(self):
        """Initialize the model and tokenizer"""
        model_path = self.model_path
        LOG.info("Loading PAT model from %s", model_path)
        base_model = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = BertPAT.from_pretrained(
            model_path,
            combine_layer_factory=CombineByScoreAdd
        )
        self.model.to(self.device)


    def get_random_split_score(self, text):
        tokens = self.tokenizer.tokenize(text)
        first, second = random_token_split(tokens)
        tokenized1 = self.tokenizer(
            first,
            is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors="pt",
        )
        tokenized2 = self.tokenizer(
            second,
            is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
            padding='max_length'
        )
        inputs = {
            'input_ids1': tokenized1['input_ids'],
            'attention_mask1': tokenized1['attention_mask'],
            'input_ids2': tokenized2['input_ids'],
            'attention_mask2': tokenized2['attention_mask'],
            'labels': torch.tensor([1]),
        }
        for k, v in inputs.items():
            v.to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, return_dict=True)

        def extract_prob_and_logit(logits):
            """Helper function to extract probability and logit values."""
            probs = torch.softmax(logits, dim=-1)[0, 1]
            logit = logits[0]
            return probs.cpu().numpy().tolist(), logit.cpu().numpy().tolist()

        # Extract probabilities and logits for all three outputs
        probs1, logits1 = extract_prob_and_logit(outputs.logits1)
        probs2, logits2 = extract_prob_and_logit(outputs.logits2)
        probs, logits = extract_prob_and_logit(outputs.logits)

        return {
            "text1": first,
            "text2": second,
            "probs1": probs1,
            "probs2": probs2,
            "probs": probs,
            "logits1": logits1,
            "logits2": logits2,
            "logits": logits
        }

def infer_tokens(sb="TwoXChromosomes"):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)
    n_item = 100
    with JobContext(model_name + "_train"):
        train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))
        train_dataset = train_dataset.take(n_item)
        model_path = get_model_save_path(model_name)
        pat = PatInference(model_path)

        for example in train_dataset:
            text = example['text']
            ret = pat.get_random_split_score(text)

            pred = ret["probs"] > 0.5
            if int(pred) != example["label"]:
                print("wrong")
            print("label", example["label"])
            for i in [1, 2]:
                k = f"text{i}"
                s=  " ".join(ret[k])
                print(f"{k}: {s}")
                for k in ["logits", "probs"]:
                    print(ret[f"{k}{i}"])

            print("probs", ret["probs"])
            print("")


if __name__ == "__main__":
    fire.Fire(infer_tokens)
