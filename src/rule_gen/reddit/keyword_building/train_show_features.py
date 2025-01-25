import os

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif, SelectKBest, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import numpy as np
import fire
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.inf_keyword_to_text import load_keyword_statement, load_train_first_100


def train_classifier(sb,  test_size=0.2, random_state=42):
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}

    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)

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
        X, y, test_size=test_size, random_state=random_state
    )
    # selector = SelectKBest(score_func=mutual_info_classif, k=n_features)
    # X_train_selected = selector.fit_transform(X_train, y_train)
    # X_test_selected = selector.transform(X_test)
    #
    # clf =  RandomForestClassifier(
    #         max_depth=5, n_estimators=10, max_features=1, random_state=42
    #     )
    clf = LogisticRegression()
    clf.fit(X_train, y_train)
    print("\nTop selected features:")
    if RandomForestClassifier == clf:
        selector = select_for_random_forest(X_train, clf, keyword_statement)
        selected_features = selector.get_support()
    else:
        features = []
        coef = clf.coef_[0]
        print(coef)
        for idx, v in enumerate(coef):
            keyword, statement = keyword_statement[idx]
            score = coef[idx]
            features.append((keyword, statement, score))
        features.sort(key=lambda x: x[2], reverse=True)
        for k_idx, (keyword, statement, score) in enumerate(features):
            print(f"Feature {k_idx}: {keyword} (Score: {score:.4f})")
            print(f"Statement: {statement}\n")
        selector = None

    y_pred = clf.predict(X_test)
    score = f1_score(y_test, y_pred)
    print(sb, score)
    run_name= f"keyword_based_{sb}_1"
    metric = "f1"
    return clf, selector


def select_for_random_forest(X_train, clf, keyword_statement):
    selector = SelectFromModel(clf, prefit=True)
    X_new = selector.transform(X_train)
    print("Selected shaped: {}".format(X_new.shape))
    print(selector.estimator_.coef_)
    selected_features = selector.get_support()
    print("\nTop selected features:")
    features = []
    for idx, is_selected in enumerate(selected_features):
        if is_selected:
            keyword, statement = keyword_statement[idx]
            score = selector.estimator._coef[idx]
            features.append((keyword, statement, score))
    features.sort(key=lambda x: x[2], reverse=True)
    for k_idx, (keyword, statement, score) in enumerate(features):
        print(f"Feature {k_idx}: {keyword} (Score: {score:.4f})")
        print(f"Statement: {statement}\n")
    return selector


def main(sb):
    # Train classifier with feature selection
    clf, selector = train_classifier(sb, )
    # You can save the model and selector here if needed
    # For example:
    # import joblib
    # joblib.dump(clf, f'classifier_{sb}.joblib')
    # joblib.dump(selector, f'selector_{sb}.joblib')


if __name__ == "__main__":
    fire.Fire(main)