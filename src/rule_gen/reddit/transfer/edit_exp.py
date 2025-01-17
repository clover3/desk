import logging
from abc import ABC, abstractmethod
from typing import List, Tuple

import numpy as np

from chair.list_lib import right
from desk_util.clf_util import load_csv_dataset_w_label, eval_prec_recall_f1_acc
from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.task_proxy import get_task_manager_proxy
LOG = logging.getLogger(__name__)


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
        editor_cls, sb,
        run_name="",
        do_pre_eval=False,
        do_report=False):
    edit_payload: list[tuple[str, int]] = load_edit_payload(sb)
    eval_payload: list[tuple[str, int]] = load_eval_payload(sb)
    edit_name = f"{sb}"

    def do_eval(payload):
        preds = editor.predict(payload)
        labels: list[int] = right(payload)
        return eval_prec_recall_f1_acc(labels, preds)

    editor = editor_cls(sb)
    if do_pre_eval:
        ret = do_eval(edit_payload)
        LOG.info("Pre-update")
        LOG.info("         : acc/f1/p/r")
        LOG.info("Train Set: %s", get_afpr(ret))

    editor.batch_edit(edit_payload, edit_name)
    ret = do_eval(edit_payload)
    LOG.info("         : acc/f1/p/r")
    LOG.info("Train Set: %s", get_afpr(ret))
    ret = do_eval(eval_payload)
    LOG.info("Val Set: %s", get_afpr(ret))


    if do_report:
        proxy = get_task_manager_proxy()
        metric = "f1"
        proxy.report_number(run_name, float(ret[metric]), "edit", metric)

