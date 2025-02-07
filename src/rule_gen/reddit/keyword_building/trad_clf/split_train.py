import os

import fire
from sklearn.tree import DecisionTreeClassifier

from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from rule_gen.cpath import output_root_path
from rule_gen.reddit.rule_classifier.common import build_feature_matrix_from_indice_paired


def train_classifier(sb, test_size=0.2, random_state=42):
    train_data = load_train_first_100(sb)
    entail_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_chatgpt3_to_text_100", f"{sb}.csv")
    X, y = build_feature_matrix_from_indice_paired(train_data, entail_save_path)
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