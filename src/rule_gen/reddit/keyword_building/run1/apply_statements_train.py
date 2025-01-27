import os

import fire

from chair.list_lib import left
from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_keyword_statement, load_train_first_100, \
    apply_statement


def main(sb):
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)
    texts = left(data)
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    make_parent_exists(res_save_path)
    apply_statement(keyword_statement, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)