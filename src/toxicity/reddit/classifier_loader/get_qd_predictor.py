import os

from omegaconf import OmegaConf

from toxicity.path_helper import get_model_save_path
from toxicity.reddit.classifier_loader.torch_misc import get_device
from toxicity.reddit.colbert.cb_inf import QDPredictor
from toxicity.reddit.colbert.modeling import get_arch_class, ColA
from toxicity.reddit.colbert.query_builders import get_sb_to_query


def get_qd_predictor(run_name):
    # colbert/churning
    model_name, sb = run_name.split("/")
    model_path = get_model_save_path(model_name)
    tokens = model_name.split("-")
    sb_strategy = "-".join(tokens[1:])
    sb_to_query = get_sb_to_query(sb_strategy)
    predictor = QDPredictor(ColA, model_path, get_device())

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

    predictor = QDPredictor(ColA, model_path, get_device())

    def predict(text):
        query = "Please classify this text"
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score

    return predict


def get_qd_predictor_w_conf(run_name):
    model_name, sb = run_name.split("/")
    conf_path = os.path.join("confs", "col", f"{model_name}.yaml")
    conf = OmegaConf.load(conf_path)
    model_path = get_model_save_path(conf.run_name)
    sb_to_query = get_sb_to_query(conf.sb_strategy)
    arch_class = get_arch_class(conf.arch_name)
    predictor = QDPredictor(arch_class, model_path, get_device())

    def predict(text):
        query = sb_to_query(sb)
        document = text
        score = predictor.predict(query, document)
        label = int(score > 0.5)
        return label, score
    return predict

