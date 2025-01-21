import logging
import random

import fire

from desk_util.io_helper import read_csv, init_logging_rivanna
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.prompt_opt.enum_wrong import enum_text_joined
from rule_gen.reddit.transfer.bert_transfer import BertTransfer2
from rule_gen.reddit.transfer.edit_exp import run_edit_exp



def load_chatgpt3_pred_on_train(sb) -> list[tuple[str, int]]:
    run_name = f"chatgpt_auto_chatgpt3_{sb}"
    dataset = f"{sb}_2_train_100"
    text_pred = [(e["text"], e["prediction"]) for e in enum_text_joined(dataset, run_name)]
    return text_pred


def main(do_report=True, sb = "SuicideWatch", do_inf_save=False):
    init_logging_rivanna()
    data = load_chatgpt3_pred_on_train(sb)
    editor_cls = lambda : BertTransfer2(data)
    run_name = sb + "_bt3"
    run_edit_exp(
        editor_cls, sb,
        run_name,
        do_report=do_report,
        do_inf_save=do_inf_save)


if __name__ == "__main__":
    fire.Fire(main)
