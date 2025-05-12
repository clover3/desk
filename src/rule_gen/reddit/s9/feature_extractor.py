import numpy as np

from nltk.tokenize import word_tokenize
import fire
from sklearn.metrics import confusion_matrix

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.path_helper import load_csv_dataset
from desk_util.runnable.run_eval import load_labels, clf_eval

import nltk
from nltk import ngrams
from collections import Counter

from rule_gen.reddit.single_run2.info_gain import information_gain

# nltk.download('punkt')


def extract_ngram_features(text, st=1, ed=10):
    list_n = list(range(st, ed+1))
    tokens = nltk.word_tokenize(text.lower())
    all_ngram_counts = Counter()

    for n in list_n:
        n_grams = list(ngrams(tokens, n))
        all_ngram_counts.update(n_grams)

    # Convert tuple keys to readable strings
    feature_dict = {tuple(gram): count for gram, count in all_ngram_counts.items()}

    return Counter(feature_dict)



def get_value(y_true, y_pred):
    try:
        m = confusion_matrix(y_true, y_pred)
        fp, tp = m[:, 1]
    except ValueError:
        print(y_true, y_pred)
        raise
    return tp / (tp + fp), tp



def main(sb= "TwoXChromosomes"):
    train_dataset = f"{sb}_2_train_100"
    min_df = 10

    data = load_csv_dataset(train_dataset)
    labels = load_labels(train_dataset)

    text_list = right(data)
    y_true = right(labels)

    featured: list[dict[tuple, int]] = list(map(extract_ngram_features, text_list))


    df = Counter()
    for d in featured:
        for key in d:
            df[key] += 1

    selected_features = {k for k, v in df.items() if v >= min_df}

    print("From {} features selected {}".format(len(df), len(selected_features)))

    table = []
    for f in selected_features:
        X: list[int] = [t[f] for t in featured]
        X = [1 if x > 0 else 0 for x in X]
        gain, _ = information_gain(np.expand_dims(X, 1), y_true)
        row = [f, get_value(y_true, X), gain]
        table.append(row)
    table.sort(key=lambda x: x[2], reverse=True)

    print_table(table[:100])



if __name__ == "__main__":
    fire.Fire(main)