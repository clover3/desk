import numpy as np




def build(j):
    j['label']
    j['tokens']
    mask = j["attention_mask"]
    scores = j["probe_score"]
    scores = np.array(mask) * np.array(scores)
    rank = scores.argsort()
    