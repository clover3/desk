import os

import fire
import numpy as np
from sklearn.tree import DecisionTreeClassifier

from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100, \
    apply_statement
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chair.list_lib import right
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.trad_clf.common import build_feature_matrix


def train_classifier(sb, test_size=0.2, random_state=42):
    keyword_statement = load_keyword_statement(sb)
    print("keyword_statement", len(keyword_statement))
    statements = right(keyword_statement)
    print("statements", statements)
    train_data = load_train_first_100(sb)
    entail_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_chatgpt3_to_text_100", f"{sb}.csv")
    X, y = build_feature_matrix(train_data, entail_save_path)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    clf = DecisionTreeClassifier(max_depth=5, random_state=42)
    clf.fit(X_train, y_train)
    score = f1_score(y_train, clf.predict(X_train))
    print("train f1", score)

    score = f1_score(y_test, clf.predict(X_test))
    print("Eval f1:", score)


def main(sb):
    train_classifier(sb)


if __name__ == "__main__":
    fire.Fire(main)
