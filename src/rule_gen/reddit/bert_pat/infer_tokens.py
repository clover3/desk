import time

import torch
import logging

import fire
from transformers import AutoTokenizer

from chair.misc_lib import Averager
from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.partition_util import get_non_sharp_indices
from rule_gen.reddit.bert_pat.pat_modeling import BertPAT, CombineByScoreAdd, BertPatFirst
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
        tokens = self.tokenizer.tokenize(text)
        if len(tokens) < window_size:
            LOG.info("Text is shorter than the window size. defaulting to full sequence")
            result = {
                'sequence': text,
                'score': self.get_full_text_score(text),
            }
            return [result]

        candidate_indices = get_non_sharp_indices(tokens)
        ret = []
        for i in range(len(candidate_indices) - window_size):
            start_idx = candidate_indices[i]
            end_idx = candidate_indices[i + window_size]
            sub_seq: list[str] = tokens[start_idx:end_idx]
            sub_text = self.tokenizer.convert_tokens_to_string(sub_seq)
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
    n_item = 100
    with JobContext(model_name + "_train"):
        train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))
        train_dataset = train_dataset.take(n_item)

        model_path = get_model_save_path(model_name)
        pat = PatInferenceFirst(model_path)
        best_evidence = []
        for example in train_dataset:
            text = example['text']
            full_score = pat.get_full_text_score(text)
            if full_score < t - 0.1:
                continue

            window_size = 5
            max_score = 0
            best_seq = None
            LOG.info("Text: %s", text)
            while max_score < t and window_size < 20:
                ret = pat.get_first_view_scores(text, window_size=window_size)
                LOG.debug("Trying window size %d", window_size)
                for result in ret:
                    score = result["score"]
                    if score > max_score:
                        max_score = score
                        best_seq = result["sequence"]

                window_size += 1
                s_outs = ["{0} ({1:.2f})".format(result['sequence'], result["score"]) for result in ret]
                LOG.debug(" ".join(s_outs))
                LOG.debug("Best seq at window={}: {} ({})".format(
                    window_size, best_seq, round(float(max_score), 2)))

            LOG.info("Best seq: {} ({})".format(best_seq, round(float(max_score), 2)))
            LOG.info("")
            best_evidence.append(best_seq)

        print("{} items".format(len(best_evidence)))
        print("After set {}".format(len(set(best_evidence))))


if __name__ == "__main__":
    fire.Fire(infer_tokens)
