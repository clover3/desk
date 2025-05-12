import numpy as np

from rule_gen.reddit.path_helper import load_j_res


def load_llg_out_style_features(dataset, run_name, seq):
    j = load_j_res(run_name, dataset)
    X = []
    for data_id, pred, score, ret_text in j:
        if ret_text == "safe":
            x = [0] * len(seq)
            pass
        else:
            safe_unsafe, codes_str = ret_text.split("\n")
            codes = codes_str.split(",")
            x = []
            for s in seq:
                if s in codes:
                    x.append(1)
                else:
                    x.append(0)
        X.append(x)
    return np.array(X)


def load_llg_out_style_features_w_score(dataset, run_name, seq):
    j = load_j_res(run_name, dataset)
    X = []
    for data_id, pred, score, ret_text in j:
        if ret_text == "safe":
            codes = []
        else:
            safe_unsafe, codes_str = ret_text.split("\n")
            codes = codes_str.split(",")
        x = []
        for s in seq:
            if s in codes:
                x.append(score)
            else:
                x.append(-score)
        X.append(x)
    return np.array(X)


def load_s9_out_style_features(dataset, run_name, convert_option, seq):
    def convert_score(score, term):
        if term == "Yes" or term == "yes":
            return score
        else:
            return -3

    def convert_score_discrete(score, term):
        if term == "Yes" or term == "yes":
            return 1
        else:
            return 0

    if convert_option == "discrete":
        convert_score_fn = convert_score_discrete
    else:
        convert_score_fn = convert_score

    j = load_j_res(run_name, dataset)
    X = []
    for data_id, _, ret_text, output in j:
        d = {}
        for code, term, score in output:
            prob = convert_score_fn(score, term)
            d[code] = prob
        try:
            x = [d[item] for item in seq]
        except KeyError:
            x = [0] * len(seq)
        X.append(x)
    return np.array(X)


def load_s9(dataset, run_name, convert_option="discrete") -> np.array:
    seq = [f"S{i}" for i in range(1, 10)]
    return load_s9_out_style_features(dataset, run_name, convert_option, seq)



def load_llg_toxic(dataset, run_name) -> np.array:
    seq = [f"S1"]
    return load_llg_out_style_features_w_score(dataset, run_name, seq)


def load_llg_default(dataset, run_name) -> np.array:
    seq = [f"S{i}" for i in range(1, 13)]
    return load_llg_out_style_features_w_score(dataset, run_name, seq)

