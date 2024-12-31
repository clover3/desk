import os
from sklearn.feature_selection import mutual_info_classif, SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import fire
from sklearn.svm import SVC

from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.inf_keyword_to_text import load_keyword_statement, load_train_first_100


def train_classifier(sb, n_features=10, test_size=0.2, random_state=42):
    """
    Train a classifier with feature selection using mutual information.

    Args:
        sb: Input parameter from main function
        n_features: Number of top features to select
        test_size: Proportion of dataset to use for testing
        random_state: Random seed for reproducibility
    """
    # Load data (using your existing code)
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)

    # Create label dictionary
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}

    # Load results
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)

    # Create feature dictionary
    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}

    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res])
    print("Use {} keywords".format(max_k_idx))
    # Prepare features and labels
    X = []
    y = []
    for t_idx in range(len(data)):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
        y.append(label_d[t_idx])

    X = np.array(X)
    y = np.array(y)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Feature selection
    selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    X_train_selected = selector.fit_transform(X_train, y_train)
    X_test_selected = selector.transform(X_test)

    # Get selected feature indices and their scores
    selected_features = selector.get_support()
    feature_scores = selector.scores_

    # Print top features and their scores
    print("\nTop selected features:")
    for idx, (is_selected, score) in enumerate(zip(selected_features, feature_scores)):
        if is_selected:
            keyword, statement = keyword_statement[idx]
            print(f"Feature {idx}: {keyword} (Score: {score:.4f})")
            print(f"Statement: {statement}\n")

    # Train classifier
    clf = SVC(kernel="linear", C=0.025, random_state=42)
    # clf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clf.fit(X_train_selected, y_train)

    # Make predictions and evaluate
    y_pred = clf.predict(X_test_selected)

    # Print classification report
    # print("\nClassification Report:")
    score = f1_score(y_test, y_pred)
    print(sb, score)
    run_name= f"keyword_based_{sb}_1"
    metric = "f1"
    # proxy = get_task_manager_proxy()
    # dataset = f"{sb}_train_holdout"
    # proxy.report_number(run_name, score, dataset, metric)

    return clf, selector


def main(sb):
    # Train classifier with feature selection
    clf, selector = train_classifier(sb, 20)
    # You can save the model and selector here if needed
    # For example:
    # import joblib
    # joblib.dump(clf, f'classifier_{sb}.joblib')
    # joblib.dump(selector, f'selector_{sb}.joblib')


if __name__ == "__main__":
    fire.Fire(main)