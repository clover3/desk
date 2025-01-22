import logging
from abc import ABC, abstractmethod
from typing import List, Tuple
import datasets
import numpy as np

from chair.list_lib import right
from desk_util.clf_util import load_csv_dataset_w_label, eval_prec_recall_f1_acc, clf_predict_w_predict_fn, \
    clf_predict_w_predict_list_fn
from desk_util.io_helper import read_csv
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
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
    data_name = "train_data2"
    p = get_reddit_train_data_path_ex(data_name, subreddit, role)
    all_data = read_csv(p)
    all_data: list[tuple[str, int]] = [(text, int(label)) for text, label in all_data]
    return all_data[:n_item]


def run_edit_exp(
        editor_factory, sb,
        run_name="",
        do_pre_eval=False,
        do_inf_save=False,
        do_report=False,
        eval_data_name=""
    ):
    if not eval_data_name:
        eval_data_name = f"{sb}_val_100"
    datasets.disable_progress_bars()
    edit_payload: list[tuple[str, int]] = load_edit_payload(sb)
    eval_payload: list[tuple[str, int]] = load_csv_dataset_w_label(eval_data_name)
    edit_name = f"{sb}"

    def do_eval(payload):
        preds = editor.predict(payload)
        labels: list[int] = right(payload)
        return eval_prec_recall_f1_acc(labels, preds)


    LOG.info("Initialize the model")
    editor = editor_factory()
    if do_pre_eval:
        ret = do_eval(edit_payload)
        LOG.info("         : acc/f1/p/r")
        LOG.info("Train Set: %s", get_afpr(ret))

    LOG.info("Updating the model")
    editor.batch_edit(edit_payload, edit_name)
    LOG.info("Runing post update model")
    ret = do_eval(edit_payload)
    LOG.info("         : acc/f1/p/r")
    LOG.info("Train Set: %s", get_afpr(ret))
    ret = do_eval(eval_payload)
    LOG.info("Val Set: %s", get_afpr(ret))
    if do_inf_save:
        dataset = eval_data_name
        clf_predict_w_predict_list_fn(dataset, run_name, editor.predict)

        if do_report:
            run_eval_clf(run_name, dataset, do_report=do_report)

