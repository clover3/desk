import os

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
import fire

from chair.list_lib import right
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from rule_gen.reddit.keyword_building.path_helper import load_keyword_statement
from rule_gen.reddit.rule_classifier.common import build_feature_matrix_from_indice_paired, build_feature_from_indices_paired


def train_classifier(sb):
    train_data = load_train_first_100(sb)
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "k_to_text_100", f"{sb}.csv")
    data_len = len(train_data)
    X_train, y_train = build_feature_matrix_from_indice_paired(train_data, entail_save_path)

    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_train)
    score = f1_score(y_train, y_pred)
    print("train f1", score)


    dataset = f"{sb}_val_100"
    payload = load_csv_dataset_by_name(dataset)
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "st_val_100", f"{sb}.csv")
    X_test = build_feature_from_indices_paired(len(payload), entail_save_path)
    y_pred = clf.predict(X_test)

    labels = load_labels(dataset)
    y_test = right(labels)
    score = f1_score(y_test, y_pred)
    print("Eval f1:", score)

    keyword_statement = load_keyword_statement(sb)
    print("keyword_statement", len(keyword_statement))
    statements = right(keyword_statement)
    print("statements", statements)


def main(sb="NeutralPolitics"):
    train_classifier(sb, )


if __name__ == "__main__":
    fire.Fire(main)