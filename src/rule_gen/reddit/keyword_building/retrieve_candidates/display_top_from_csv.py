import os
import random
from collections import defaultdict

import fire

from chair.list_lib import right, left
from chair.misc_lib import average
from desk_util.io_helper import read_csv, init_logging
from llama_user.llama_helper.llama_model_names import Llama3_8B_Instruct
from rule_gen.cpath import output_root_path
from rule_gen.reddit.decode.llama_prob import LlamaProbabilityCalculator
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from rule_gen.reddit.keyword_building.path_helper import load_keyword_statement
from rule_gen.reddit.keyword_building.retrieve_candidates.precision_at_k import precision_at_ks


def main(sb="NeutralPolitics"):
    init_logging()
    keyword_statement = load_keyword_statement(sb)
    keywords = left(keyword_statement)
    statements = right(keyword_statement)
    data = load_train_first_100(sb)

    gpt_res_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    gpt_res = read_csv(gpt_res_path)

    est_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100_res", f"{sb}.csv")
    est = read_csv(est_save_path)
    offset = 0

    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in gpt_res}
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in gpt_res]) + 1

    perf_method = defaultdict(list)
    perf_random = defaultdict(list)
    for t_idx in range(0, len(data)):
        y_true = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        print("Text: ", data[t_idx])
        rows = est[offset: offset + len(statements)]
        offset += len(statements)
        indices_scores = [(i, float(t[0])) for i, t in enumerate(rows)]
        indices_scores.sort(key=lambda x: x[1], reverse=True)
        y_true_indices = [k_idx for k_idx in range(max_k_idx) if y_true[k_idx]]
        sorted_indices = [e[0] for e in indices_scores]
        random_indices = list(range(max_k_idx))
        random.shuffle(random_indices)
        print("y_true_indices: ", y_true_indices)
        for k in range(10):
            i, score = indices_scores[k]
            print(keywords[i], statements[i], i, i in y_true_indices, round(score, 2))


if __name__ == "__main__":
    fire.Fire(main)
