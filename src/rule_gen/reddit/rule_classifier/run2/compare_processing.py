import os
from collections import Counter

import fire

from chair.list_lib import right, dict_value_map
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import load_csv_dataset
import ast

from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.path_helper import load_named_keyword_statement
from rule_gen.reddit.entail_helper import load_statement_appy_result


def load(entail_save_path):
    d = load_statement_appy_result(entail_save_path)
    return d


def load_probs(entail_save_path):
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    d = {(int(t_idx), int(k_idx)): ast.literal_eval(s) for k_idx, t_idx, s in res}

    entail_idx = 1
    threshold = 0.2
    d = dict_value_map(lambda x: int(x[entail_idx] > threshold), d)
    return d

def load_condnli(entail_save_path, texts, features):
    n = len(texts)
    res: list[tuple[str, str, str]] = read_csv(entail_save_path)
    data = [int(float(e[0]) > 0.5) for e in res]
    d = {}
    i = 0
    for t_idx in range(n):
        for k_idx, feature in enumerate(features):
            d[t_idx, k_idx] = data[i]
            i += 1
    return d



def main(sb):
    name = "chatgpt3"
    entail_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_chatgpt3_to_text_100", f"{sb}.csv")
    run1 = load(entail_save_path)
    # entail_save_path = os.path.join(output_root_path, "reddit",
    #                              "rule_processing", "k_chatgpt3_nli_to_text_100", f"{sb}.csv")
    # run2 = load_probs(entail_save_path)
    keyword_statement = load_named_keyword_statement(name, sb)
    features = right(keyword_statement)
    dataset = f"{sb}_val_100"
    texts = right(load_csv_dataset(dataset))
    entail_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100_res", f"{sb}.csv")
    run3 = load_condnli(entail_save_path, texts, features)

    agree = Counter()
    true_1 = Counter()
    true_2 = Counter()
    for t_idx, t in enumerate(texts):
        print("Text: ", t)
        table = []
        for k_idx, feature in enumerate(features):
            true_1[k_idx, run1[t_idx, k_idx]] += 1
            true_2[k_idx, run3[t_idx, k_idx]] += 1

            if run1[t_idx, k_idx] != run3[t_idx, k_idx]:
                row = [run1[t_idx, k_idx], run3[t_idx, k_idx], feature]
                table.append(row)
            else:
                agree[k_idx] += 1
        print_table(table)

    for k_idx, cnt in agree.most_common():
        print(cnt, true_1[k_idx, 1], true_2[k_idx, 1], features[k_idx])


if __name__ == "__main__":
    fire.Fire(main)