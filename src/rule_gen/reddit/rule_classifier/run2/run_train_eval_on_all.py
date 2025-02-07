import numpy as np
import os
from dataclasses import dataclass
from typing import Tuple, Optional

import fire
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.tree import DecisionTreeClassifier, export_text

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.path_helper import load_clf_pred
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import (
    load_train_first_100,
)
from rule_gen.reddit.keyword_building.path_helper import load_named_keyword_statement
from rule_gen.reddit.rule_classifier.common import build_feature_matrix_from_indice_paired, build_feature_from_indices_paired
from rule_gen.reddit.path_helper import get_split_subreddit_list


def add_new_feature(X, new_feature):
    x0 = np.reshape(new_feature, [-1, 1])
    X = np.concatenate((x0, X), axis=1)
    return X


class FeatureLoaderFromPaired:
    def __init__(self, subreddit: str,
                 augment_other=False,
                 ):
        self.subreddit = subreddit
        self.keyword_name = "chatgpt3"
        self.augment_other = augment_other
        if augment_other:
            print("Use feature augmentation")

    def _get_data_path(self, data_type: str) -> str:
        folder = f"k_{self.keyword_name}_to_text_100" \
            if data_type == "train" else f"k_{self.keyword_name}_to_text_val_100"
        return os.path.join(
            output_root_path,
            "reddit",
            "rule_processing",
            folder,
            f"{self.subreddit}.csv"
        )

    def load_train_data(self) -> Tuple[np.array, np.array]:
        """Load and prepare training data."""
        train_data = load_train_first_100(self.subreddit)
        entail_save_path = self._get_data_path("train")
        X, y = build_feature_matrix_from_indice_paired(train_data, entail_save_path)
        if self.augment_other:
            y_as_x = load_other_pred(self.subreddit, "train")
            X = add_new_feature(X, y_as_x)
        return X, y


    def load_val_data(self) -> Tuple[np.array, np.array]:
        """Load and prepare validation data."""
        dataset = f"{self.subreddit}_val_100"
        payload = load_csv_dataset_by_name(dataset)
        entail_save_path = self._get_data_path("val")
        X_test = build_feature_from_indices_paired(len(payload), entail_save_path)
        labels = load_labels(dataset)
        if self.augment_other:
            y_as_x = load_other_pred(self.subreddit, "val")
            X_test = add_new_feature(X_test, y_as_x)
        return X_test, right(labels)


def load_other_pred(sb, split):
    dataset = f"{sb}_{split}_100"
    run_name = f"chatgpt_none"
    preds = load_clf_pred(dataset, run_name)
    ys = [e[1] for e in preds]
    return ys


def check_data_availability(subreddit: str) -> bool:
    """Check if both training and validation data exists for a subreddit."""
    return all(
        os.path.exists(path) for path in [
            os.path.join(output_root_path, "reddit", "rule_processing", folder, f"{subreddit}.csv")
            for folder in ["k_chatgpt3_to_text_100", "k_chatgpt3_to_text_val_100"]
        ]
    )


def main(augment_other=False):
    table = []
    for subreddit in get_split_subreddit_list("train"):
        if not check_data_availability(subreddit):
            continue

        try:
            loader = FeatureLoaderFromPaired(
                subreddit,
                augment_other=augment_other
            )
            X_train, y_train = loader.load_train_data()
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_score = f1_score(y_train, train_pred)
            # Validate
            X_test, y_test = loader.load_val_data()
            val_pred = clf.predict(X_test)
            val_score = f1_score(y_test, val_pred)
            # Generate tree representation
            keyword_statement = load_named_keyword_statement(loader.keyword_name, loader.subreddit)
            statements = right(keyword_statement)
            if loader.augment_other:
                statements = ["<ChatGPT None>"] + statements
            row = [subreddit, val_score]
            table.append(row)

            print(f"\nResults for subreddit: {subreddit}")
            print(f"Training F1 Score: {train_score:.4f}")
            print(f"Validation F1 Score: {val_score:.4f}")

        except ValueError as e:
            print(f"Error processing subreddit {subreddit}: {str(e)}")
            continue
        except KeyError as e:
            print(f"Error processing subreddit {subreddit}: {str(e)}")
            continue
    print_table(table)


if __name__ == "__main__":
    fire.Fire(main)