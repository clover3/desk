import tqdm
import os
from abc import ABC, abstractmethod
from typing import List

import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from chair.list_lib import right, left
from chair.misc_lib import get_second, get_first
from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_clf_pred_save_path


def eval_prec_recall_f1_acc(y_true: List[int], y_pred: List[int]) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": len(y_true)
    }


def print_evaluation_results(metrics: dict):
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1 Score: {metrics['f1_score']:.4f}")
    print("Confusion Matrix:")
    print(np.array(metrics['confusion_matrix']))


class BinaryDataset(ABC):
    # Has key id, text, label
    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def __getitem__(self, idx):
        pass


def clf_predict_w_predict_fn(dataset, run_name, predict_fn):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    def predict(e):
        id, text = e
        label, score = predict_fn(text)
        return id, label, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path = get_clf_pred_save_path(run_name, dataset)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def clf_predict_w_batch_predict_fn(dataset, run_name, batch_predict_fn):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    ids = left(payload)
    ls_iter = batch_predict_fn(right(payload))

    def pred_itr():
        for data_id, (label, score) in zip(ids, ls_iter):
            yield data_id, label, score

    pred_itr = tqdm(pred_itr(), desc="Processing", unit="item", total=len(payload))
    save_path = get_clf_pred_save_path(run_name, dataset)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")
