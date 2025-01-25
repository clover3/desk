import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import os
from tabulate import tabulate

from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.inf_keyword_to_text import load_keyword_statement, load_train_first_100


def load_reddit_data(sb):
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)
    label_d = {t_idx: int(label) for t_idx, (_text, label) in enumerate(data)}

    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)

    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}

    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res])
    X = []
    y = []
    for t_idx in range(len(data)):
        features = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        X.append(features)
        y.append(label_d[t_idx])

    return np.array(X), np.array(y)


def compare_classifiers(subreddits, test_size=0.2, random_state=42):
    names = [
        "Nearest Neighbors",
        "Linear SVM",
        "RBF SVM",
        "Gaussian Process",
        "Decision Tree",
        "Random Forest",
        "Neural Net",
        "AdaBoost",
        "Naive Bayes",
        "QDA",
    ]

    classifiers = [
        KNeighborsClassifier(3),
        SVC(kernel="linear", C=0.025, random_state=42),
        SVC(gamma=2, C=1, random_state=42),
        GaussianProcessClassifier(1.0 * RBF(1.0), random_state=42),
        DecisionTreeClassifier(max_depth=5, random_state=42),
        RandomForestClassifier(
            max_depth=5, n_estimators=10, max_features=1, random_state=42
        ),
        MLPClassifier(alpha=1, max_iter=1000, random_state=42),
        AdaBoostClassifier(random_state=42),
        GaussianNB(),
        QuadraticDiscriminantAnalysis(),
    ]

    results = []
    for sb in subreddits:
        print(f"Processing subreddit: {sb}")
        X, y = load_reddit_data(sb)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        scores = []
        for name, clf in zip(names, classifiers):
            pipeline = make_pipeline(StandardScaler(), clf)
            pipeline.fit(X_train, y_train)
            y_pred = pipeline.predict(X_test)
            score = f1_score(y_test, y_pred)
            scores.append(score)

        results.append([sb] + scores)

    headers = ["Subreddit"] + names
    print("\nF1 Scores:")
    print(tabulate(results, headers=headers, floatfmt=".3f", tablefmt="grid"))

    output_dir = os.path.join(output_root_path, "reddit", "classifier_comparison")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "comparison_results.txt")

    with open(output_path, 'w') as f:
        f.write(tabulate(results, headers=headers, floatfmt=".3f", tablefmt="grid"))
    print(f"\nSaved results to: {output_path}")


def main(subreddits):
    if isinstance(subreddits, str):
        subreddits = [s.strip() for s in subreddits.split(',')]
    compare_classifiers(subreddits)


if __name__ == "__main__":
    import fire
    fire.Fire(main)