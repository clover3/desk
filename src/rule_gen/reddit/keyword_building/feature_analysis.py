import os
from sklearn.feature_selection import mutual_info_classif

import fire

from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.inf_keyword_to_text import load_keyword_statement, load_train_first_100


def main(sb):
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)

    label_d = {}
    for t_idx, (_text, label) in enumerate(data):
        label_d[t_idx] = label
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)
    d = {}
    for k_idx, t_idx, ret in res:
        pred = {"True": 1, "False": 0}[ret]
        d[(int(t_idx), int(k_idx))] = pred

    X = []
    y = []
    for t_idx in range(len(data)):
        features = []
        for k_idx in range(len(keyword_statement)):
            features.append(d[(t_idx, k_idx)])
        X.append(features)
        y.append(label_d[t_idx])

    ret = mutual_info_classif(X, y)
    print(ret)
    # table = []
    # for k_idx, (k, s) in e    numerate(keyword_statement):
    #     true = counter[k_idx, "True"]
    #     false = counter[k_idx, "False"]
    #     print(k_idx)
    #     row = [k_idx, true / (true + false), true, false, k, s]
    #     table.append(row)
    #
    # table.sort(key=lambda x: x[1], reverse=True)
    #
    # for row in table:
    #     print(row)


if __name__ == "__main__":
    fire.Fire(main)
