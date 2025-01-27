import os

import fire
import numpy as np
from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100, \
    apply_statement
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chair.list_lib import right
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path


def train_classifier(sb, test_size=0.2, random_state=42):
    keyword_statement = load_keyword_statement(sb)
    print("keyword_statement", len(keyword_statement))
    statements = right(keyword_statement)
    print("statements", statements)
    data = load_train_first_100(sb)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}

    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)
    print("{} lines".format(len(res)))

    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}

    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res])
    X = []
    y = []
    for t_idx in range(len(data)):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
        y.append(label_d[t_idx])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    f1_list = []
    for feature_i in range(max_k_idx):
        rep = X[:, feature_i]
        y_pred = np.array(rep > 0).astype(int)
        score = f1_score(y, y_pred)
        f1_list.append((score, keyword_statement[feature_i]))
    f1_list.sort(key=lambda x: x[0], reverse=True)

    for s, k in f1_list[:3]:
        print(s, k)


def main(sb):
    train_classifier(sb, )


if __name__ == "__main__":
    fire.Fire(main)
