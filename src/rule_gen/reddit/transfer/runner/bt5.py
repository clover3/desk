import fire

from desk_util.io_helper import init_logging_rivanna
from rule_gen.reddit.transfer.bert_transfer import BertTransferBalance, build_training_argument
from rule_gen.reddit.transfer.edit_exp import run_edit_exp
from rule_gen.reddit.transfer.runner.bt2_100 import load_train_mix3_n_item
from rule_gen.reddit.transfer.runner.bt3 import load_chatgpt3_pred_on_train



def main(
    learning_rate: float = 5e-5,
    num_train_epochs: int = 3,
    do_report: bool = True,
    sb: str = "SuicideWatch",
    do_inf_save: bool = False
):
    init_logging_rivanna()
    train_data2 = load_train_mix3_n_item(100)
    train_arg = build_training_argument(
        logging_dir="",
        output_dir="",
        learning_rate=learning_rate,
        num_train_epochs=num_train_epochs
    )
    chatgpt3_gen_data = load_chatgpt3_pred_on_train(sb)
    data = train_data2 + chatgpt3_gen_data
    editor_cls = lambda: BertTransferBalance(train_arg, data)
    run_name = sb + f"_BTB_{learning_rate}_{num_train_epochs}"
    run_edit_exp(
        editor_cls, sb,
        run_name,
        do_report=do_report,
        do_inf_save=do_inf_save)


if __name__ == "__main__":
    fire.Fire(main)
