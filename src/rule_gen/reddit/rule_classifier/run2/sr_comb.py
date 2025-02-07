from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

from chair.tab_print import print_table
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_n_rules
from rule_gen.reddit.rule_classifier.clf_feature_loading import load_dataset_from_predictions


def main():
    table = []
    for sb in get_split_subreddit_list("train"):
        try:
            n_rule = get_n_rules(sb)
            run_name_iter = [f"chatgpt_sr_{sb}_{idx}_both" for idx in range(n_rule)]
            print(sb)

            dataset = f"{sb}_val_100"
            X, y = load_dataset_from_predictions(run_name_iter, dataset)
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
            continue
        except FileNotFoundError as e:
            print(e)
    print_table(table)




if __name__ == "__main__":
    main()