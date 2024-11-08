import os
from omegaconf import OmegaConf

import torch
from transformers import pipeline
from toxicity.path_helper import get_model_save_path
from toxicity.reddit.colbert.cb_inf import QDPredictor
from toxicity.reddit.colbert.query_builders import get_sb_to_query


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


def get_clf_pipeline_w_q(run_name):
    model_name, sb = run_name.split("/")
    model_path = get_model_save_path(model_name)
    tokens = model_name.split("-")
    sb_strategy = "-".join(tokens[1:])
    sb_to_query = get_sb_to_query(sb_strategy)
    predictor = QDPredictor(model_path, get_device())

    def predict(text):
        query = sb_to_query(sb)
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score
    return predict


def get_clf_pipeline_w_conf(run_name):
    model_name, sb = run_name.split("/")
    conf_path = os.path.join("confs", "col", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    model_path = get_model_save_path(model_name)
    sb_to_query = get_sb_to_query(conf.sb_strategy)
    predictor = QDPredictor(model_path, get_device())

    def predict(text):
        query = sb_to_query(sb)
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score
    return predict


def get_colbert_const(run_name):
    model_name = run_name
    model_path = get_model_save_path(model_name)

    predictor = QDPredictor(model_path, get_device())

    def predict(text):
        query = "Please classify this text"
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score

    return predict
