from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chair.list_lib import right
from chair.tab_print import print_table
from desk_util.path_helper import load_csv_dataset
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.rule_classifier.clf_feature_loading import load_dataset_from_predictions


def main():
    table = []
    name = "gnq2"
    run_name_iter = [f"chatgpt_v2_{name}_{idx}" for idx in range(18)]
    for sb in get_split_subreddit_list("train"):
        print(sb)
        try:
            dataset = f"{sb}_2_val_100"
            X, y = load_dataset_from_predictions(run_name_iter, dataset)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            payload = load_csv_dataset(dataset)
            texts = right(payload)
            clf = LogisticRegression()
            clf.fit(X_train, y_train)
            train_pred = clf.predict(X_train)
            train_score = f1_score(y_train, train_pred)

            for i, (y_p, y_g) in enumerate(zip(train_pred, y_train)):
                if y_p != y_g:
                    text = texts[i]
                    print(y_g, y_p, text)

            print(f"\nResults for subreddit: {sb}")
            print(f"Training F1 Score: {train_score:.4f}")

        except (ValueError, KeyError) as e:
            print(f"Error processing subreddit {sb}: {str(e)}")
            continue
        except FileNotFoundError as e:
            print(e)
            break
    print_table(table)



if __name__ == "__main__":
    main()