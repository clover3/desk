import matplotlib.pyplot as plt
import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import fire
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text

from chair.list_lib import right
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100


def train_classifier(sb,  test_size=0.2, random_state=42):
    keyword_statement = load_keyword_statement(sb)
    print("keyword_statement", len(keyword_statement))
    statements = right(keyword_statement)
    print("statements", statements)
    data = load_train_first_100(sb)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "k_to_text_100", f"{sb}.csv")
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    print("{} lines".format(len(res)))
    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}
    print("Feature sparsity = {}".format(sum(d.values()) / len(d)))
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res])
    print("Use {} keywords".format(max_k_idx))
    X = []
    y = []
    for t_idx in range(len(data)):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
        y.append(label_d[t_idx])

    X = np.array(X)
    y = np.array(y)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state)
    clf = DecisionTreeClassifier(max_depth=1, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred)
    text_representation = export_text(clf, feature_names=statements[:-1])
    print(text_representation)
    print(sb, score)


def main(sb):
    train_classifier(sb, )


if __name__ == "__main__":
    fire.Fire(main)