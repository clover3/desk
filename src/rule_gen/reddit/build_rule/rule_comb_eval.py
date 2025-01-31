import json

from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import f1_score

from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_rule_path, get_n_rules
from desk_util.runnable.run_eval import load_labels


def rule_comb(sb):
    n_rule = get_n_rules(sb)
    print(sb)

    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))

    dataset = f"{sb}_val_100"
    labels = load_labels(dataset)
    labels_d = dict(labels)

    rows = []
    preds_list = []
    for rule_idx in range(n_rule):
        run_name = f"chatgpt_sr_{sb}_{rule_idx}_both"
        preds = load_clf_pred(dataset, run_name)
        preds_list.append(preds)

    X_list = []
    for rule_idx in range(n_rule):
        preds = preds_list[rule_idx]
        X = []
        Y = []
        for data_id, pred, _ in preds:
            Y.append(labels_d[data_id])
            X.append([pred])
        X_list.append(X)
        mi = mutual_info_classif(X, Y, discrete_features=True)
        row = [rule_idx, mi[0], rules[rule_idx]]
        rows.append(row)

    rows.sort(key=lambda x: x[1], reverse=True)

    for n_used in range(1, n_rule + 1):
        rule_indices = [e[0] for e in rows[:n_used]]
        n_data = len(preds_list[0])
        new_preds = []
        for d_i in range(n_data):
            features = []
            for r_i in rule_indices:
                _data_id, pred, _ = preds_list[r_i][d_i]
                features.append(pred)
            new_pred = any(features)
            new_preds.append(new_pred)

        y_gold = []
        for data_id, _pred, _ in preds_list[0]:
            y_gold.append(labels_d[data_id])

        score = f1_score(y_gold, new_preds)
        print(n_used, round(score, 2))


def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            rule_comb(sb)
        except (FileNotFoundError, ValueError) as e:
            print(e)



if __name__ == "__main__":
    main()
