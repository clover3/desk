import os
from typing import Union, List

import torch
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class QDPredictor:
    def __init__(self, arch_class, model_path: str, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        base_model = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = arch_class.from_pretrained(model_path)
        self.model.colbert_set_up(self.tokenizer)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 512

    def preprocess(self, query: str, document: str):
        query_encoding = self.tokenizer(
            query,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        doc_encoding = self.tokenizer(
            document,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'query_input_ids': query_encoding['input_ids'].to(self.device),
            'query_attention_mask': query_encoding['attention_mask'].to(self.device),
            'doc_input_ids': doc_encoding['input_ids'].to(self.device),
            'doc_attention_mask': doc_encoding['attention_mask'].to(self.device)
        }

    def predict(self, query: str, document: str) -> Union[float, List[float]]:
        inputs = self.preprocess(query, document)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.cpu().numpy()
            return float(scores[0][0])
