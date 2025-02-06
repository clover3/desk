import json
import os

import fire

from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_statement_path, load_keyword_statement
from rule_gen.reddit.keyword_building.trad_clf.train_show_features import train_w_statements


def train_classifier(sb, test_size=0.2, random_state=42):
    name = "chatgpt3"
    keyword_statement = json.load(open(get_named_keyword_statement_path(name, sb), "r"))

    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", f"k_{name}_to_text_100", f"{sb}.csv")

    return train_w_statements(keyword_statement, random_state, res_save_path, sb, test_size)


if __name__ == "__main__":
    fire.Fire(train_classifier)