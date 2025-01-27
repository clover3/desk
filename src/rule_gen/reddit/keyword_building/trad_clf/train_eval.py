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
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100


def build_dataset(data, entail_save_path):
    data_len = len(data)
    X = build_feature(data_len, entail_save_path)

    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}
    y = []
    for t_idx in range(len(data)):
        y.append(label_d[t_idx])
    return X, y


def build_feature(data_len, entail_save_path):
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    print("{} lines".format(len(res)))
    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}
    print("Feature sparsity = {}".format(sum(d.values()) / len(d)))
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res])
    print("Use {} keywords".format(max_k_idx))
    X = []
    for t_idx in range(data_len):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
    return X


def train_classifier(sb):
    train_data = load_train_first_100(sb)
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "k_to_text_100", f"{sb}.csv")
    X_train, y_train = build_dataset(train_data, entail_save_path)


    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    score = f1_score(y_train, y_pred)
    print("train f1", score)


    dataset = f"{sb}_val_100"
    payload = load_csv_dataset_by_name(dataset)
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "st_val_100", f"{sb}.csv")
    X_test = build_feature(len(payload), entail_save_path)
    y_pred = clf.predict(X_test)


    labels = load_labels(dataset)
    y_test = right(labels)
    score = f1_score(y_test, y_pred)
    print("Eval f1:", score)

    keyword_statement = load_keyword_statement(sb)
    print("keyword_statement", len(keyword_statement))
    statements = right(keyword_statement)
    print("statements", statements)
    text_representation = export_text(clf, feature_names=statements[:-1])
    print(text_representation)


def main(sb):
    train_classifier(sb, )


if __name__ == "__main__":
    fire.Fire(main)