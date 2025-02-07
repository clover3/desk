import json
import os

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import numpy as np

from chair.list_lib import right
from desk_util.path_helper import load_csv_dataset
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.rule_classifier.clf_feature_loading import load_dataset_from_predictions



def analyze_instance_features(clf, X, instance_idx, feature_names=None, top_k=5):
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(X.shape[1])]

    coeffs = clf.coef_[0]
    instance = X[instance_idx]
    contributions = coeffs * instance
    feature_importance = list(zip(feature_names, contributions, instance))
    sorted_features = sorted(feature_importance,
                             key=lambda x: abs(x[1]),
                             reverse=True)

    return sorted_features[:top_k]


def print_top_features_for_instances(
        clf, X, y, y_pred, feature_names=None, n_instances=5,
        texts=None,
):
    incorrect_predictions = np.where(y != y_pred)[0]

    print("\nAnalyzing feature importance for incorrect predictions:")
    for idx in incorrect_predictions[:n_instances]:
        print(f"\nInstance {idx}")
        print("Text: ", texts[idx])
        print(f"True label: {y[idx]}, Predicted: {y_pred[idx]}")

        top_features = analyze_instance_features(clf, X, idx, feature_names)
        print("Top contributing features:")
        for feature, contribution, value in top_features:
            print(f"{feature:20} | Contribution: {contribution:8.4f} | Value: {value:8.4f}")


def load_features_names():
    file_name = "gnq2"
    rule_save_path = os.path.join(
        output_root_path, "reddit", "rule_processing", f"{file_name}.json")
    rules = json.load(open(rule_save_path, "r"))
    return rules


# Add to your main function:
def main():
    name = "gnq2"
    n_sub_item = 18
    features = load_features_names()
    features_short = [" ".join(t.split(" ")[:2]) for t in features]
    for i, t in enumerate(features):
        print(i, t)
    print("----")
    for sb in get_split_subreddit_list("train"):
        print(sb)
        run_name_iter = [f"chatgpt_v2_{name}_{idx}" for idx in range(n_sub_item)]
        try:
            dataset = f"{sb}_2_train_100"
            texts = right(load_csv_dataset(dataset))
            X, y = load_dataset_from_predictions(run_name_iter, dataset)
            X[:, 10] = 0
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)

            # Analyze feature importance
            print_top_features_for_instances(clf, X_train, y_train, train_pred, n_instances=100,
                                             feature_names=features_short, texts=texts)
        except (ValueError, KeyError) as e:
            print(f"Error processing subreddit {sb}: {str(e)}")
            continue
        except FileNotFoundError as e:
            print(e)
            break

        break


if __name__ == '__main__':
    main()