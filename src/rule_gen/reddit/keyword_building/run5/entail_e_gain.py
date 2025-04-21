import json

import fire
import numpy as np
from sklearn.metrics import confusion_matrix
from rule_gen.cpath import output_root_path
import os
from chair.list_lib import right
from chair.tab_print import print_table
from rule_gen.reddit.entail_helper import load_statement_appy_result_as_table
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from rule_gen.reddit.keyword_building.path_helper import get_entail_save_path, get_statements_from_ngram
from rule_gen.reddit.single_run2.info_gain import information_gain


def main(sb):
    name = "ts"
    table = []
    statement_path = os.path.join(
        output_root_path, "reddit",
        "ngram_based_j2", f"{sb}.json")

    feature_table: list[list[int]] = load_statement_appy_result_as_table(
        get_entail_save_path(name, sb))
    print(feature_table)
    statements = json.load(open(statement_path, "r"))
    data = load_train_first_100(sb)
    labels = right(data)
    y_true = list(map(int, labels))

    for k_idx, feature in enumerate(feature_table):
        y_pred = np.array(feature).reshape([-1, 1])
        gain, d = information_gain(y_pred, y_true)
        row = [k_idx, round(gain, 2), confusion_matrix(y_true, y_pred).tolist(), statements[k_idx]]
        table.append(row)
    print_table(table)


if __name__ == "__main__":
    fire.Fire(main)
