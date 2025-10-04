import logging

import torch
from transformers import AutoTokenizer

from rule_gen.reddit.bert_pat.pat_modeling import CombineByScoreAdd, BertPatFirst

LOG = logging.getLogger("InfPat")


class PatInferenceFirst:
    def __init__(self, model_path, debug=False, window_size=1):
        self.debug = debug
        self.model_path = model_path
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
            sub_text: str = " ".join(sp_tokens[i: i + window_size])
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
                'range': (i, i + window_size),
                'score': probs1[0, 1].cpu().numpy().tolist(),
            }
            ret.append(result)
        return ret
