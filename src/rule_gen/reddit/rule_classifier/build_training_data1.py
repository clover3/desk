import json
import os
import random
from typing import Iterable
from chair.misc_lib import group_by, get_first
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import load_csv_dataset
from chair.list_lib import right
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_statement_path


def enum_training_data() -> Iterable[tuple[str, str, int]]:
    n_pos = 2
    n_neg = 2
    name = "chatgpt3"
    sb_list = ["churning", "nba", "news", "pokemon", "pokemongo"]
    for sb in sb_list:
        entail_save_path = os.path.join(output_root_path, "reddit",
                                        "rule_processing", f"k_{name}_to_text_100", f"{sb}.csv")

        keyword_statement_list = json.load(open(get_named_keyword_statement_path(name, sb), "r"))
        statements = right(keyword_statement_list)

        payload = load_csv_dataset(f"{sb}_val_100")
        texts = right(payload)

        res: list[tuple[str, str, str]] = read_csv(entail_save_path)
        res_parsed = [(int(k_idx), int(t_idx), {"True": 1, "False": 0}[ret]) for k_idx, t_idx, ret in res]
        g = group_by(res_parsed, get_first)
        for k_idx, entries in g.items():
            query_like = statements[k_idx]
            pos = [(t_idx, label) for _k_idx, t_idx, label in entries if label]
            neg = [(t_idx, label) for _k_idx, t_idx, label in entries if not label]
            pos_sel = random.sample(pos, n_pos) if n_pos < len(pos) else pos
            neg_sel = random.sample(neg, n_neg) if n_neg < len(neg) else neg

            for t_idx, label in pos_sel + neg_sel:
                doc_like = texts[t_idx]
                yield query_like, doc_like, label


def main():
    entail_train_data_path = os.path.join(output_root_path, "reddit",
                                    "rule_processing", f"entail_training.csv")
    save_csv(enum_training_data(), entail_train_data_path)



if __name__ == "__main__":
    main()