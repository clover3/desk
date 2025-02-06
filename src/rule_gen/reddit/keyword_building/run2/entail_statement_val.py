import json
import os

import fire

from chair.list_lib import left, right
from chair.misc_lib import make_parent_exists
from desk_util.clf_util import load_csv_dataset_by_name
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_statement_path, load_keyword_statement
from taskman_client.wrapper3 import JobContext


def main(sb):
    name = "chatgpt3"
    run_name = f"k_{name}_to_text_val_100_{sb}"
    with JobContext(run_name):
        res_save_path = os.path.join(output_root_path, "reddit",
                                     "rule_processing", f"k_{name}_to_text_val_100", f"{sb}.csv")
        keyword_statement = json.load(open(get_named_keyword_statement_path(name, sb), "r"))
        dataset = f"{sb}_val_100"
        payload = load_csv_dataset_by_name(dataset)
        texts = right(payload)
        make_parent_exists(res_save_path)
        apply_statement(keyword_statement, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)
