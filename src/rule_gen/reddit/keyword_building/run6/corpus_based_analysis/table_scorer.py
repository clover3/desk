import numpy as np
import pickle

from nltk import ngrams

from chair.misc_lib import average
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_terms
from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.term_norm_match import get_bert_basic_tokenizer
from rule_gen.reddit.path_helper import get_rp_path


def get_table_clf(run_name):
    _, sb = run_name.split('/')
    n_st = 2
    n_ed = 6
    n_list = list(range(n_st, n_ed))
    print("Loading data...")
    dir_name = "run6_10k_score"
    score_d: dict[int, np.array] = {}
    term_dd: dict[int, dict[str, int]] = {}
    for n in n_list:
        score_path = get_rp_path(dir_name, f"{sb}.{n}.pkl")
        scores = pickle.load(open(score_path, "rb"))
        score_d[n] = scores
        term_list: list[tuple | str] = load_run6_10k_terms(n)
        if n == 1:
            term_list = [(t, ) for t in term_list]

        d = {term: i for i, term in enumerate(term_list)}
        term_dd[n] = d

    tokenize_fn = get_bert_basic_tokenizer()
    t = 0.5
    print("Use average., T=", t)
    def predict(text):
        tokens = tokenize_fn(text)
        scores = []
        ngram_list = []
        for n in range(n_st, n_ed):
            d = term_dd[n]
            for seq in ngrams(tokens, n):
                if seq in d:
                    idx = d[seq]
                    score = score_d[n][idx]
                    scores.append(score)
                    ngram_list.append(seq)

        print("tokens,scores", len(tokens), len(scores))
        out_s_l = ["{0}:{1:.2f}".format(t, s) for t, s in zip(ngram_list, scores)]
        out_s = " ".join(out_s_l[:30])
        print(out_s)
        if scores:
            score = average(scores)
        else:
            score = 0
        return int(score > t), score
    return predict