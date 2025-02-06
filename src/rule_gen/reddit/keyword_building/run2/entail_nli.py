import csv
import json
import os
import fire
from chair.list_lib import left, apply_batch
from chair.misc_lib import make_parent_exists, TimeEstimator
from rule_gen.cpath import output_root_path
from rule_gen.reddit.keyword_building.apply_statement_common import load_train_first_100
from rule_gen.reddit.keyword_building.path_helper import get_named_keyword_statement_path, load_keyword_statement
from rule_gen.reddit.keyword_building.retrieve_candidates.nli_roberta import NLIProcessor


def apply_statement(keyword_statement, res_save_path, texts):
    n_req = len(keyword_statement) * len(texts)
    proc = NLIProcessor(batch_size=16)
    out_f = open(res_save_path, "w")
    csv_writer = csv.writer(out_f)
    ticker = TimeEstimator(n_req, sample_size=110)
    for k_idx, ks in enumerate(keyword_statement):
        keyword, statement = ks
        for t_idx, text in enumerate(texts):
            probs = proc.process_pairs([text, statement])[0].tolist()
            row = [k_idx, t_idx, probs]
            csv_writer.writerow(row)
            ticker.tick()


def main(sb):
    name = "chatgpt3"
    keyword_statement = json.load(open(get_named_keyword_statement_path(name, sb), "r"))
    data = load_train_first_100(sb)
    texts = left(data)

    res_name = name + "_nli"
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", f"k_{res_name}_to_text_100", f"{sb}.csv")
    make_parent_exists(res_save_path)
    apply_statement(keyword_statement, res_save_path, texts)


if __name__ == "__main__":
    fire.Fire(main)
