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
from rule_gen.reddit.rule_classifier.common import build_feature_matrix, build_feature
from rule_gen.reddit.path_helper import get_split_subreddit_list


@dataclass
class ModelConfig:
    max_depth: int = 5
    random_state: int = 42

def add_new_feature(X, new_feature):
    x0 = np.reshape(new_feature, [-1, 1])
    X = np.concatenate((x0, X), axis=1)
    return X


class SubredditClassifier:
    def __init__(self, subreddit: str,
                 augment_other=False,
                 config: Optional[ModelConfig] = None):
        self.subreddit = subreddit
        self.config = config or ModelConfig()
        self.keyword_name = "chatgpt3"
        # self.clf = Logi(
        #     max_depth=self.config.max_depth,
        #     random_state=self.config.random_state
        # )
        self.clf = LogisticRegression()
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

    def _load_train_data(self) -> Tuple[np.array, np.array]:
        """Load and prepare training data."""
        train_data = load_train_first_100(self.subreddit)
        entail_save_path = self._get_data_path("train")
        X, y = build_feature_matrix(train_data, entail_save_path)
        if self.augment_other:
            y_as_x = load_other_pred(self.subreddit, "train")
            X = add_new_feature(X, y_as_x)
        return X, y


    def _load_val_data(self) -> Tuple[np.array, np.array]:
        """Load and prepare validation data."""
        dataset = f"{self.subreddit}_val_100"
        payload = load_csv_dataset_by_name(dataset)
        entail_save_path = self._get_data_path("val")
        X_test = build_feature(len(payload), entail_save_path)
        labels = load_labels(dataset)
        if self.augment_other:
            y_as_x = load_other_pred(self.subreddit, "val")
            X_test = add_new_feature(X_test, y_as_x)
        return X_test, right(labels)

    def train_and_evaluate(self) -> Tuple[float, float]:
        X_train, y_train = self._load_train_data()
        self.clf.fit(X_train, y_train)
        train_pred = self.clf.predict(X_train)
        train_score = f1_score(y_train, train_pred)

        # Validate
        X_test, y_test = self._load_val_data()
        val_pred = self.clf.predict(X_test)
        val_score = f1_score(y_test, val_pred)

        # Generate tree representation
        keyword_statement = load_named_keyword_statement(self.keyword_name, self.subreddit)
        statements = right(keyword_statement)
        if self.augment_other:
            statements = ["<ChatGPT None>"] + statements

        return train_score, val_score


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
            classifier = SubredditClassifier(
                subreddit,
                augment_other=augment_other
            )
            train_score, val_score = classifier.train_and_evaluate()
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