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
from rule_gen.reddit.bert_pat.partition_util import get_non_sharp_indices
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
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = BertPAT.from_pretrained(
            model_path,
            combine_layer_factory=CombineByScoreAdd
        )
        self.model.to(self.device)


    def get_first_view_scores(self, text, window_size=None):
        if window_size is None:
            window_size = self.window_size
        tokens = self.tokenizer.tokenize(text)
        candidate_indices = get_non_sharp_indices(tokens)
        ret = []
        for i in range(len(candidate_indices) - window_size):
            start_idx = candidate_indices[i]
            end_idx = candidate_indices[i + window_size]
            sub_seq: list[str] = tokens[start_idx:end_idx]
            sub_text = self.tokenizer.convert_tokens_to_string(sub_seq)
            encoded, inputs = self.encode_input(sub_seq)
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            probs1 = torch.softmax(outputs.logits1, -1)
            result = {
                'sequence': sub_text,
                'input_ids': encoded['input_ids'][0].cpu().numpy(),
                'score': probs1[0, 1].cpu().numpy(),
            }
            ret.append(result)
        return ret


    def encode_input(self, sub_seq):
        encoded = self.tokenizer(
            sub_seq,
            is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {
            'input_ids1': encoded['input_ids'].to(self.device),
            'attention_mask1': encoded['attention_mask'].to(self.device),
            'input_ids2': encoded['input_ids'].to(self.device),
            'attention_mask2': encoded['attention_mask'].to(self.device),
            'labels': torch.tensor([1], device=self.device)
        }
        return encoded, inputs


class PatInferenceFirst:
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
        base_model = 'bert-base-uncased'
        LOG.info("Loading PAT model from %s", model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = BertPatFirst.from_pretrained(
            model_path,
            combine_layer_factory=CombineByScoreAdd
        )
        self.model.to(self.device)

    def get_full_text_score(self, text):
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {
            'input_ids1': encoded['input_ids'].to(self.device),
            'attention_mask1': encoded['attention_mask'].to(self.device),
        }
        with torch.no_grad():
            logits1 = self.model(**inputs, return_dict=True)

        probs1 = torch.softmax(logits1, -1)
        return probs1[0, 1].cpu().numpy()

    def get_first_view_scores(self, text, window_size=None):
        if window_size is None:
            window_size = self.window_size

        sp_tokens = text.split()
        if len(sp_tokens) < window_size:
            LOG.debug("Text is shorter than the window size. defaulting to full sequence")
            result = {
                'sequence': text,
                'score': self.get_full_text_score(text),
            }
            return [result]

        ret = []
        for i in range(len(sp_tokens) - window_size):
            sub_text: str = " ".join(sp_tokens[i: i+window_size])
            encoded = self.tokenizer(
                sub_text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt'
            )
            inputs = {
                'input_ids1': encoded['input_ids'].to(self.device),
                'attention_mask1': encoded['attention_mask'].to(self.device),
            }
            with torch.no_grad():
                logits1 = self.model(**inputs, return_dict=True)

            probs1 = torch.softmax(logits1, -1)
            result = {
                'sequence': sub_text,
                'input_ids': encoded['input_ids'][0].cpu().numpy(),
                'score': probs1[0, 1].cpu().numpy(),
            }
            ret.append(result)
        return ret



def infer_tokens(sb="TwoXChromosomes"):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)

    t = 0.8
    t_strong = 0.9
    n_item = 100
    confusion = Counter()
    print("High scoring tokens in negative examples")
    with JobContext(model_name + "_train"):
        train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))
        train_dataset = train_dataset.take(n_item)
        bert_tms = get_classifier("bert2_train_mix_sel")

        model_path = get_model_save_path(model_name)
        pat = PatInferenceFirst(model_path)
        for example in train_dataset:
            text = example['text']
            tms_pred, tms_score = bert_tms(text)
            full_score = pat.get_full_text_score(text)
            domain_pred = int(full_score > 0.5)
            confusion[(tms_pred, domain_pred)] += 1
            if full_score > 0.4:
                continue

            n_grams = []
            for window_size in range(1, 4):
                ret = pat.get_first_view_scores(text, window_size=window_size)
                for result in ret:
                    score = result["score"]
                    if score > t_strong:
                        max_score = score
                        # print(result["sequence"], score)
                        n_grams.append((result["sequence"], score, 2))
            if n_grams:
                print("Text", text)
                print("label", example["label"])
                print(n_grams)

if __name__ == "__main__":
    fire.Fire(infer_tokens)
