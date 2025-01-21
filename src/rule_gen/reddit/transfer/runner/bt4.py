import fire

from desk_util.io_helper import init_logging_rivanna
from rule_gen.reddit.transfer.bert_transfer import BertTransfer2
from rule_gen.reddit.transfer.edit_exp import run_edit_exp
from rule_gen.reddit.transfer.runner.bt2_100 import load_train_data2
from rule_gen.reddit.transfer.runner.bt3 import load_chatgpt3_pred_on_train


def main(do_report=True, sb="SuicideWatch", do_inf_save=False):
    init_logging_rivanna()
    train_data2 = load_train_data2(100)
    chatgpt3_gen_data = load_chatgpt3_pred_on_train(sb)
    data = train_data2 + chatgpt3_gen_data
    editor_cls = lambda: BertTransfer2(data)
    run_name = sb + "_bt4"
    run_edit_exp(
        editor_cls, sb,
        run_name,
        do_report=do_report,
        do_inf_save=do_inf_save)


if __name__ == "__main__":
    fire.Fire(main)
