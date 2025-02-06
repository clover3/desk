import ast
import os
import fire
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.path_helper import load_keyword_statement
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from rule_gen.cpath import output_root_path



def build_feature(data_len, entail_save_path):
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    entail_idx = 1
    print("{} lines".format(len(res)))
    print("Real value")
    d = {(int(t_idx), int(k_idx)): ast.literal_eval(s)[entail_idx]  for k_idx, t_idx, s in res}
    # print("Feature sparsity = {}".format(sum(d.values()) / len(d)))
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res]) + 1
    print("Use {} keywords".format(max_k_idx))
    X = []
    for t_idx in range(data_len):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
    return X


def build_feature_matrix(paired_data: list[tuple[str, str]], entail_save_path):
    data_len = len(paired_data)
    X = build_feature(data_len, entail_save_path)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(paired_data)}
    y = []
    for t_idx in range(data_len):
        y.append(label_d[t_idx])
    return X, y


def train_classifier(sb, test_size=0.2, random_state=42):
    train_data = load_train_first_100(sb)
    entail_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_chatgpt3_nli_to_text_100", f"{sb}.csv")
    X, y = build_feature_matrix(train_data, entail_save_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    # clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    score = f1_score(y_train, clf.predict(X_train))
    print("train f1", score)

    score = f1_score(y_test, clf.predict(X_test))
    print("Eval f1:", score)


def main(sb):
    train_classifier(sb)


if __name__ == "__main__":
    fire.Fire(main)
