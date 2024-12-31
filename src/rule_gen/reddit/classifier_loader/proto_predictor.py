from desk_util.path_helper import get_model_save_path
import os
from typing import Union, List

import torch
from transformers import AutoTokenizer

from rule_gen.reddit.proto.protory_net2 import ProtoryNet3

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ProtoPredictor:
    def __init__(self, arch_class, model_path: str, device: str = None):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        base_model = 'bert-base-uncased'

        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = arch_class.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = 512

    def preprocess(self, text: str):
        encoding = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].to(self.device),
            'attention_mask': encoding['attention_mask'].to(self.device)
        }

    def predict(self, document: str) -> Union[float, List[float]]:
        inputs = self.preprocess(document)
        with torch.no_grad():
            proto_out = self.model(**inputs)
            probs = torch.sigmoid(proto_out.logits)
            scores = probs.cpu().numpy()
            return float(scores[0][0])


def get_proto_predictor(run_name):
    model_path = get_model_save_path(run_name)
    predictor = ProtoPredictor(ProtoryNet3, model_path)

    def predict(text):
        score = predictor.predict(text)
        label = int(score > 0.5)
        return label, score
    return predict

