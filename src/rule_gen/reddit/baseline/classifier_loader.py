import tqdm
import os

from krovetzstemmer import Stemmer
from nltk import word_tokenize

from rule_gen.cpath import output_root_path
from rule_gen.reddit.baseline.ngram_logit import NgramLogisticRegression


def get_ngram_logit_classifiers(run_name):
    tokens = run_name.split("_")
    sb = "_".join(tokens[1:])
    if tokens[0] == "ngramlogit":
        dir_name = "ngram_logit"
    elif tokens[0] == "ngramlogit4":
        dir_name = "ngram_logit4"
    else:
        print("First token {} is not expected".format(tokens[0]))
        raise ValueError()
    model_path = os.path.join(output_root_path, "models", dir_name, f"{sb}.pickle")
    model = NgramLogisticRegression.load_model(model_path)
    stemmer = Stemmer()

    def predict(text):
        tokens = word_tokenize(text)
        tokens = [stemmer(t) for t in tokens]
        pred = model.predict([(tokens, 0)])[0]
        score = model.predict_proba([(tokens, 0)])[0, 1]
        return pred, score

    return predict


