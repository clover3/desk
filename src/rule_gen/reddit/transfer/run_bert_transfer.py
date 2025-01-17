import logging

from rule_gen.reddit.transfer.runner.bert_transfer import BertTransfer
from rule_gen.reddit.transfer.edit_exp import run_edit_exp


def main():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    editor_cls = BertTransfer
    sb = "SuicideWatch"
    run_edit_exp(editor_cls, sb)


if __name__ == "__main__":
    main()
