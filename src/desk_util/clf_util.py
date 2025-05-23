import os
from typing import List, Callable

import numpy as np
import tqdm
from tqdm import tqdm

from chair.list_lib import right, left
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_clf_pred_save_path, get_label_path


def eval_prec_recall_f1_acc(y_true: List[int], y_pred: List[int]) -> dict:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
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


def clf_predict_w_predict_fn(
        dataset, run_name,
        predict_fn: Callable[[str], tuple[int, float]],
        overwrite_existing=False,
):
    payload = load_csv_dataset_by_name(dataset)

    def predict(e):
        id, text = e
        label, score = predict_fn(text)
        return id, label, score

    save_path = get_clf_pred_save_path(run_name, dataset)
    if not overwrite_existing and os.path.exists(save_path):
        if len(read_csv(save_path)) == len(payload):
            print(f"Prediction exists. Skip prediction")
            print(f": {save_path}")
            return
        else:
            print(f"Prediction exists but not complete. Overwritting")
            print(f": {save_path}")

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def clf_predict_w_predict_list_fn(dataset, run_name,
                                  predict_list_fn: Callable[[list[str]], list[int]]):
    payload = load_csv_dataset_by_name(dataset)
    text_list = right(payload)
    pred_itr = predict_list_fn(text_list)

    save_payload = []
    for id, p in zip(left(payload), pred_itr):
        save_payload.append((id, p, 0))

    save_path = get_clf_pred_save_path(run_name, dataset)
    save_csv(save_payload, save_path)
    print(f"Saved at {save_path}")


def load_csv_dataset_by_name(dataset):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)
    return payload


def load_csv_dataset_w_label(dataset):
    payload = load_csv_dataset_by_name(dataset)
    labels = read_csv(get_label_path(dataset))
    output = []
    for (data_id1, text), (data_id2, label_s) in zip(payload, labels):
        assert data_id1 == data_id2
        output.append((text, int(label_s)))
    return output


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
