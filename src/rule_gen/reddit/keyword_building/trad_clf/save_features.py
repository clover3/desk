import pandas as pd
import os
from sklearn.feature_selection import mutual_info_classif

import fire

from chair.list_lib import right
from chair.misc_lib import make_parent_exists
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.trad_clf.train_eval import build_feature


def save_using_df(X, y, res_save_path):
    df = pd.DataFrame(X)
    df['label'] = y
    make_parent_exists(res_save_path)
    df.to_csv(res_save_path, index=False)




def build_feature_matrix(per_feature_res, n_feature, label_d):
    d = {}
    for k_idx, t_idx, ret in per_feature_res:
        pred = {"True": 1, "False": 0}[ret]
        d[(int(t_idx), int(k_idx))] = pred
    X = []
    y = []
    for t_idx in range(n_feature):
        features = []
        for k_idx in range(n_feature):
            features.append(d[(t_idx, k_idx)])
        X.append(features)
        y.append(label_d[t_idx])
    return X, y


def main(sb):
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)
    label_d = {}
    for t_idx, (_text, label) in enumerate(data):
        label_d[t_idx] = label
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)
    n_feature = len(keyword_statement)
    X, y = build_feature_matrix(res, n_feature, label_d)
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100_table", f"{sb}.csv")
    save_using_df(X, y, res_save_path)


def save_for_val(sb):
    dataset = f"{sb}_val_100"
    payload = load_csv_dataset_by_name(dataset)
    entail_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "st_val_100", f"{sb}.csv")
    X = build_feature(len(payload), entail_save_path)
    labels = load_labels(dataset)
    y = right(labels)
    feature_save = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100_table", f"{sb}_val.csv")
    save_using_df(X, y, feature_save)



if __name__ == "__main__":
    fire.Fire(save_for_val)
