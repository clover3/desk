import logging

import torch

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_pat.infer_tokens import PatInference, PatInferenceFirst

LOG = logging.getLogger("InfPat")


class PatBasedClassifier(PatInferenceFirst):
    def __init__(self, model_path, debug=False, window_size=1):
        super(PatBasedClassifier, self).__init__(model_path, debug=debug, window_size=window_size)

    def get_window_logits_list(self, text, window_size_list):
        sp_tokens = text.split()
        range_list = []
        for window_size in window_size_list:
            for i in range(len(sp_tokens) - window_size):
                range_list.append((i, i + window_size))
        logits_list = []
        if not range_list:
            if len(sp_tokens) > min(window_size_list):
                print("Something wrong")
                raise ValueError()
            range_list.append((0, len(sp_tokens)))

        for st, ed in range_list:
            sub_text: str = " ".join(sp_tokens[st: ed])
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
                logits = self.model(**inputs, return_dict=True)
                logits_list.append(logits)
        return logits_list

    def classify(self, text, strategy, window_size_list=[1, 3, 5]):
        if strategy == "max":
            logits_list = self.get_window_logits_list(text, window_size_list)
            probs = []
            for logit in logits_list:
                prob = torch.softmax(logit, -1)[0, 1]
                probs.append(prob)
            max_prob = torch.max(torch.tensor(probs)).cpu().numpy()
            return float(max_prob)
        elif strategy == "logit_avg":
            logits_list = self.get_window_logits_list(text, window_size_list)
            logits = torch.stack(logits_list, axis=0)
            avg_logits = torch.mean(logits, 0)
            prob = torch.softmax(avg_logits, -1)[0, 1].cpu().numpy()
            return float(prob)
        else:
            raise ValueError(strategy)



def get_pat_predictor(run_name):
    model_path = get_model_save_path(run_name)
    pat = PatBasedClassifier(model_path)

    def predict(text):
        score = pat.classify(text, "logit_avg", [3])
        label = int(score > 0.5)
        return label, score
    return predict

