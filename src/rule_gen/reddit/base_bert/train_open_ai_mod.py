import logging

import fire

from taskman_client.task_proxy import get_task_manager_proxy
from taskman_client.wrapper3 import JobContext
from desk_util.io_helper import init_logging
from rule_gen.open_ai_mod.path_helper import get_open_ai_mod_csv_path
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.predict_clf import predict_clf_main
from rule_gen.reddit.reddit_train_bert import build_training_argument, DataArguments, finetune_bert

LOG = logging.getLogger(__name__)


def train_exp():
    init_logging()
    data_name = "open_ai_mod"
    model_name = f"bert2_{data_name}"
    with JobContext(model_name + "_train"):
        base_model = 'bert-base-uncased'

        output_dir = get_model_save_path(model_name)
        final_model_dir = get_model_save_path(model_name)
        logging_dir = get_model_log_save_dir_path(model_name)
        max_length = 256
        training_args = build_training_argument(logging_dir, output_dir)
        dataset_args = DataArguments(
            train_data_path=get_open_ai_mod_csv_path("train"),
            eval_data_path=get_open_ai_mod_csv_path("val"),
            max_length=max_length
        )
        eval_result = finetune_bert(
            model_name=base_model,
            training_args=training_args,
            dataset_args=dataset_args,
            final_model_dir=final_model_dir,
        )

        proxy = get_task_manager_proxy()
        for metric in ["eval_f1"]:
            dataset = data_name + "_val"
            metric_short = metric[len("eval_"):]
            proxy.report_number(model_name, eval_result[metric], dataset, metric_short)

        predict_clf_main(model_name, data_name + "_val", do_eval=True, do_report=True)


# Example usage:
if __name__ == "__main__":
    fire.Fire(train_exp)