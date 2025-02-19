import json
import os

import fire

from chair.list_lib import left
from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_statement_path, load_keyword_statement
from taskman_client.wrapper3 import JobContext


def main(sb):
    name = "chatgpt3"
    run_name = f"k_{name}_to_text_100_{sb}"
    with JobContext(run_name):
        res_save_path = os.path.join(output_root_path, "reddit",
                                     "rule_processing", f"k_{name}_to_text_100", f"{sb}.csv")
        keyword_statement = json.load(open(get_named_keyword_statement_path(name, sb), "r"))
        data = load_train_first_100(sb)
        texts = left(data)
        make_parent_exists(res_save_path)
        apply_statement(keyword_statement, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)
