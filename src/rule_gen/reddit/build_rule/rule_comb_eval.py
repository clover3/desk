import json

from sklearn.feature_selection import mutual_info_classif
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
    for rule_idx in range(n_rule):
        run_name = f"chatgpt_sr_{sb}_{rule_idx}_both"
        preds = load_clf_pred(dataset, run_name)

        X = []
        Y = []
        for data_id, pred, _ in preds:
            Y.append(labels_d[data_id])
            X.append([pred])

        mi = mutual_info_classif(X, Y, discrete_features=True)
        row = [rule_idx, mi[0], rules[rule_idx]]
        rows.append(row)

    rows.sort(key=lambda x: x[1], reverse=True)
    print(rows[0])



def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            rule_comb(sb)
        except FileNotFoundError:
            pass



if __name__ == "__main__":
    main()