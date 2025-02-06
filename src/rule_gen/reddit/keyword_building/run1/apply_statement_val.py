import os

import fire

from chair.list_lib import right
from chair.misc_lib import make_parent_exists
from desk_util.clf_util import load_csv_dataset_by_name
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.path_helper import load_keyword_statement


def main(sb):
    dataset = f"{sb}_val_100"
    keyword_statement = load_keyword_statement(sb)
    payload = load_csv_dataset_by_name(dataset)
    texts = right(payload)

    res_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_processing", "k_to_text_100", f"{sb}.csv")

    make_parent_exists(res_save_path)
    apply_statement(keyword_statement, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)
