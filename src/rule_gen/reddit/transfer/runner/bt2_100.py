import logging
import random

import fire

from desk_util.io_helper import read_csv, init_logging_rivanna
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.transfer.bert_transfer import BertTransfer2
from rule_gen.reddit.transfer.edit_exp import run_edit_exp



def load_train_data2(n_item):
    p = get_reddit_train_data_path_ex("train_data", "train_mix3", "train")
    all_data = read_csv(p)
    random.seed(42)
    random.shuffle(all_data)
    sel_data = all_data[:n_item]
    sel_data: list[tuple[str, int]] = [(text, int(label)) for text, label in sel_data]
    return sel_data


def main(do_report=True, sb = "SuicideWatch", do_inf_save=False):
    init_logging_rivanna()
    n_item = 100
    data = load_train_data2(n_item)
    editor_cls = lambda : BertTransfer2(data)
    run_name = sb + f"_bt2_{n_item}"
    run_edit_exp(
        editor_cls, sb,
        run_name,
        do_report=do_report,
        do_inf_save=do_inf_save)


if __name__ == "__main__":
    fire.Fire(main)
