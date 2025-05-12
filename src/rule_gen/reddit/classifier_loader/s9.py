import numpy as np
import pickle

from rule_gen.reddit.s9.s9_loader import get_s9_combined
from rule_gen.cpath import output_root_path
import os

def get_s9_per_sb(run_name):
    # llama_s9_sb
    tokens = run_name.split("_")
    sb = "_".join(tokens[2:])
    max_text_len = 5000
    get_feature = get_s9_combined()

    model_path = os.path.join(output_root_path, "models", "sklearn_run4", f"{sb}.pickle")
    clf = pickle.load(open(model_path, "rb"))

    def predict(text):
        text = text[:max_text_len]
        feature = get_feature(text)
        print(feature)
        pred = clf.predict([feature])[0]
        score = clf.predict_proba([feature])[0, 1]
        return pred, score
    return predict


def get_s9g(run_name):
    # llama_s9_sb
    max_text_len = 5000
    get_feature = get_s9_combined()

    model_path = os.path.join(output_root_path, "models", "sklearn_run5", f"all.pickle")
    clf = pickle.load(open(model_path, "rb"))
    if run_name == "llama_s9g0":
        clf.intercept_ = clf.intercept_ * 0


    def predict(text):
        text = text[:max_text_len]
        feature = get_feature(text)
        print(feature)
        pred = clf.predict([feature])[0]
        score = clf.predict_proba([feature])[0, 1]
        return pred, score
    return predict


def get_s9_classifiers(run_name):
    if run_name.startswith("llama_s9_"):
        return get_s9_per_sb(run_name)
    else:
        return get_s9g(run_name)


