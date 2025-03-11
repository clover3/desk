import os
from typing import Union, List

import torch
from omegaconf import OmegaConf
from transformers import AutoTokenizer, BertForSequenceClassification

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.classifier_loader.torch_misc import get_device
from rule_gen.reddit.colbert.modeling import get_arch_class
from rule_gen.reddit.colbert.query_builders import get_sb_to_query

os.environ["TOKENIZERS_PARALLELISM"] = "true"


class ConcatPredictor:
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

    def preprocess(self, query: str, document: str):
        encodings = self.tokenizer(
            query,
            document,
            padding='max_length',  # Changed from 'padding=True'
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        return {
            'input_ids': encodings['input_ids'].to(self.device),
            'attention_mask': encodings['attention_mask'].to(self.device),
            # 'labels': labels
        }

    def predict(self, query: str, document: str) -> Union[float, List[float]]:
        inputs = self.preprocess(query, document)
        with torch.no_grad():
            outputs = self.model(**inputs)
            scores = outputs.logits.cpu().numpy()
            return float(scores[0][1])


def get_ce_predictor_w_conf(run_name):
    # concat_emb1/churning
    prefix = "ce_"
    run_name = run_name[len(prefix):]

    model_name, sb = run_name.split("/")
    conf_path = os.path.join("confs", "cross_encoder", f"{model_name}.yaml")
    conf = OmegaConf.load(conf_path)
    model_path = get_model_save_path(conf.run_name)
    sb_to_query = get_sb_to_query(conf.sb_strategy)
    arch_class = BertForSequenceClassification
    predictor = ConcatPredictor(arch_class, model_path, get_device())

    def predict(text):
        query = sb_to_query(sb)
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score

    return predict
