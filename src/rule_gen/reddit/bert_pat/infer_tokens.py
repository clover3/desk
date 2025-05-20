import torch
import logging

import fire
from transformers import AutoTokenizer

from rule_gen.reddit.bert_pat.partition_util import get_non_sharp_indices, random_token_split
from rule_gen.reddit.bert_pat.pat_modeling import BertPAT, CombineByScoreAdd, BertPatFirst
from rule_gen.reddit.bert_pat.scratch import tokenize_and_split

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
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        except OSError:
            base_model = 'bert-base-uncased'
            self.tokenizer = AutoTokenizer.from_pretrained(base_model)

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


    def predict_bipartition(self, text):
        inputs = tokenize_and_split([text], self.tokenizer, self.max_length, "pt")
        new_inputs = {}
        for k, v in inputs.items():
            # v = torch.unsqueeze(v, 0)
            new_inputs[k] = v.to(self.device)

        new_inputs['labels'] = torch.tensor([1], device=self.device)

        with torch.no_grad():
            outputs = self.model(**new_inputs, return_dict=True)
        probs = torch.softmax(outputs.logits, -1)
        return float(probs[:, 1].cpu().numpy())




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
                'sub_text': text,
                'score': self.get_full_text_score(text).tolist(),
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
                'sub_text': sub_text,
                'input_ids': encoded['input_ids'][0].cpu().numpy().tolist(),
                'range': (i, i+window_size),
                'score': probs1[0, 1].cpu().numpy().tolist(),
            }
            ret.append(result)
        return ret

