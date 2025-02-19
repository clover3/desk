import logging
import os
import random

import fire

from chair.list_lib import right, left
from desk_util.io_helper import read_csv, init_logging
from llama_user.llama_helper.llama_model_names import Llama3_8B_Instruct
from rule_gen.cpath import output_root_path
from rule_gen.reddit.decode.llama_prob import LlamaProbabilityCalculator
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from rule_gen.reddit.keyword_building.path_helper import load_keyword_statement
from rule_gen.reddit.keyword_building.retrieve_candidates.precision_at_k import precision_at_ks, recall_at_ks


def main(sb="NeutralPolitics"):
    init_logging()
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    keyword_statement = load_keyword_statement(sb)
    statements = right(keyword_statement)
    data = load_train_first_100(sb)

    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    res = read_csv(res_save_path)

    d = {(int(t_idx), int(k_idx)): {"True": 1, "False": 0}[ret]
         for k_idx, t_idx, ret in res}
    calculator = LlamaProbabilityCalculator(Llama3_8B_Instruct)
    max_k_idx = max([int(k_idx) for k_idx, t_idx, ret in res]) + 1
    for t_idx in range(2, len(data)):
        text, label = data[t_idx]
        y_true = [d[(t_idx, k_idx)] for k_idx in range(max_k_idx)]
        prompt = f"Is this statement true for this text? Answer Yes/No\n"
        prompt += f"<text>{text}</text>"
        prompt += f"<statement>"
        prefix = prompt + ".\n "

        yn_statement = []
        for s in statements:
            yn_statement.append(s + "\n assistant: Yes")
            yn_statement.append(s + "\n assistant: No")

        # output: list[tuple[int, str, float]] = calculator.compute_probability_trie(prefix, yn_statement)
        output: list[tuple[int, str, float]] = calculator.compute_pair_probability_raw(prefix, yn_statement)

        y_true_indices = [k_idx for k_idx in range(max_k_idx) if y_true[k_idx]]
        sorted_indices = [e[0] for e in output]
        random_indices = list(range(max_k_idx))
        random.shuffle(random_indices)
        print("Pred indices", sorted_indices)
        print("y_true_indices: ", y_true_indices)
        ks = [1, 5, 10, 20, 40]  # List of k values to evaluate
        p_at_ks = recall_at_ks(y_true, sorted_indices, ks)
        print("recall_at_ks")
        print("Your pred: ", p_at_ks)
        p_at_ks = recall_at_ks(y_true, random_indices, ks)
        print("Random: ", p_at_ks)

        break


if __name__ == "__main__":
    fire.Fire(main)
