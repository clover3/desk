import fire
import numpy as np
from nltk.metrics.aline import feature_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chair.tab_print import print_table
from rule_gen.reddit.keyword_building.run3.ask_to_llama import load_feature_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.rule_classifier.clf_feature_loading import load_dataset_from_predictions


def main(sb="TwoXChromosomes"):
    table = []
    name = "gnq2"
    n_sub_item = 18
    print(sb)
    feature_run_name = f"llama_rp_cq_{sb}"
    run_name_iter = [f"chatgpt_v2_{name}_{idx}" for idx in range(n_sub_item)]
    try:
        dataset = f"{sb}_2_train_100_200"
        X1, y = load_dataset_from_predictions(run_name_iter, dataset)
        preds: list[dict] = load_feature_pred(feature_run_name, dataset)
        X2 = [t["result"] for t in preds]

        # X[:, 1] = 0
        X = np.concatenate([X1, X2], axis=1)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        clf = LogisticRegression()
        clf.fit(X_train, y_train)
        train_pred = clf.predict(X_train)

        train_score = f1_score(y_train, train_pred)
        val_pred = clf.predict(X_test)
        val_score = f1_score(y_test, val_pred)
        row = [sb, val_score]
        table.append(row)

        print(f"\nResults for subreddit: {sb}")
        print(f"Training F1 Score: {train_score:.4f}")
        print(f"Validation F1 Score: {val_score:.4f}")

    except (ValueError, KeyError) as e:
        print(f"Error processing subreddit {sb}: {str(e)}")
    except FileNotFoundError as e:
        print(e)

    print_table(table)


if __name__ == '__main__':
    fire.Fire(main)
