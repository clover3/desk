import logging
import math
import time
from abc import ABC, abstractmethod
from typing import List, Tuple
import datasets
import numpy as np
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import speed_metrics

from chair.list_lib import right
from desk_util.clf_util import load_csv_dataset_w_label, eval_prec_recall_f1_acc, clf_predict_w_predict_fn, \
    clf_predict_w_predict_list_fn
from desk_util.io_helper import read_csv, init_logging_rivanna
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.reddit.transfer.bert_transfer import BertTransfer
from taskman_client.task_proxy import get_task_manager_proxy
from transformers.utils.logging import disable_progress_bar

LOG = logging.getLogger("EDIT_EXP")


class ClfEditorSpec(ABC):
    @abstractmethod
    def batch_edit(self, edit_payload: List[Tuple[str, int]], edit_name: str = "") -> None:
        pass

    @abstractmethod
    def predict(self, eval_payload: List[Tuple[str, int]]) -> np.ndarray:
        pass


def get_afpr(ret):
    s_list = []
    for metric in ["accuracy", "f1", "precision", "recall"]:
        s_list.append("{0:.2f}".format(ret[metric]))
    return "/".join(s_list)


def load_edit_payload(subreddit):
    n_item = 10
    role = "train"
    return load_from_train_data_path(subreddit, role, n_item)


def load_eval_payload(subreddit):
    return load_csv_dataset_w_label(f"{subreddit}_val_100")


def load_from_train_data_path(subreddit, role, n_item):
    data_name = "train_data"
    p = get_reddit_train_data_path_ex(data_name, subreddit, role)
    all_data = read_csv(p)
    all_data: list[tuple[str, int]] = [(text, int(label)) for text, label in all_data]
    return all_data[:n_item]


def run_edit_exp(
        editor_factory, sb,
    ):
    datasets.disable_progress_bars()

    edit_payload: list[tuple[str, int]] = load_edit_payload(sb)
    editor = editor_factory()

    print("Init Trainer")
    output_dir = "tmp_trainer"
    args = TrainingArguments(output_dir="tmp_trainer", disable_tqdm=True)
    trainer = Trainer(model=editor.model, args=args)
    print("Init dataset")
    tokenized_eval = editor._get_tokenized_dataset(edit_payload)
    print("trainer.predict")
    disable_progress_bar()
    # raw_predictions = trainer.predict(tokenized_eval)
    self = trainer
    self._memory_tracker.start()

    test_dataloader = self.get_test_dataloader(tokenized_eval)
    start_time = time.time()
    ignore_keys = None
    metric_key_prefix = "test"
    print("one")
    output = self.evaluation_loop(
        test_dataloader, description="Prediction", ignore_keys=ignore_keys, metric_key_prefix=metric_key_prefix
    )
    print("two")
    total_batch_size = self.args.eval_batch_size * self.args.world_size
    if f"{metric_key_prefix}_jit_compilation_time" in output.metrics:
        start_time += output.metrics[f"{metric_key_prefix}_jit_compilation_time"]
    if f"{metric_key_prefix}_model_preparation_time" in output.metrics:
        start_time += output.metrics[f"{metric_key_prefix}_model_preparation_time"]
    output.metrics.update(
        speed_metrics(
            metric_key_prefix,
            start_time,
            num_samples=output.num_samples,
            num_steps=math.ceil(output.num_samples / total_batch_size),
        )
    )
    print("three")

    self.control = self.callback_handler.on_predict(self.args, self.state, self.control, output.metrics)
    print("four")
    self._memory_tracker.stop_and_update_metrics(output.metrics)


def main():
    init_logging_rivanna()
    editor_cls = BertTransfer
    run_edit_exp(editor_cls, sb = "SuicideWatch")


if __name__ == "__main__":
    main()