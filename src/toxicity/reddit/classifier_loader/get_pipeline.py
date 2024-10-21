import torch
from transformers import pipeline

from toxicity.path_helper import get_model_save_path


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_classifier_pipeline(run_name):
    model_path = get_model_save_path(run_name)
    pipe = pipeline("text-classification", model=model_path, device=get_device())

    label_map = {
        "LABEL_0": "0",
        "LABEL_1": "1",
    }

    def predict(text):
        r = pipe(text, truncation=True)[0]
        label = label_map[r["label"]]
        score = r["score"]
        if label == "0":
            score = -score
        return label, score

    def batch_predict(text_list):
        for r in pipe(text_list, truncation=True):
            label = label_map[r["label"]]
            score = r["score"]
            if label == "0":
                score = -score
            yield label, score

    return predict
