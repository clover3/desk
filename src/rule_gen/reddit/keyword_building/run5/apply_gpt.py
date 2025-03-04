import json

import fire

from chair.list_lib import left
from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100, \
    apply_statement
from rule_gen.reddit.keyword_building.path_helper import get_entail_save_path, get_statements_from_ngram
from taskman_client.wrapper3 import JobContext


def main(sb):
    name = "ts"
    run_name = f"k_{name}_to_text_100_{sb}"
    with JobContext(run_name):
        statement_path = get_statements_from_ngram(sb)
        res_save_path = get_entail_save_path(name, sb)

        statements = json.load(open(statement_path, "r"))
        data = load_train_first_100(sb)
        texts = left(data)
        make_parent_exists(res_save_path)
        apply_statement(statements, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)
