import logging
import fire

from desk_util.io_helper import init_logging_rivanna
from rule_gen.reddit.transfer.bert_transfer import BertTransfer
from rule_gen.reddit.transfer.edit_exp import run_edit_exp


def main(do_report=False, sb = "SuicideWatch", do_inf_save=False):
    init_logging_rivanna()
    editor_cls = BertTransfer
    run_name = "{}_{}".format(sb, editor_cls.editor_name)
    run_edit_exp(
        editor_cls, sb,
        run_name,
        do_report=do_report,
        do_inf_save=do_inf_save,
        eval_data_name=f"{sb}_2_val_100"
    )


if __name__ == "__main__":
    fire.Fire(main)
