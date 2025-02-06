import os
import random

import fire

from chair.list_lib import right, left
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv, init_logging, save_csv
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
    def enum_pairs():
        data = load_train_first_100(sb)
        for t_idx in range(0, len(data)):
            text, label = data[t_idx]
            for k in keywords:
                yield text, k
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100_payload", f"{sb}.csv")
    make_parent_exists(res_save_path)
    save_csv(enum_pairs(), res_save_path)


if __name__ == "__main__":
    fire.Fire(main)
