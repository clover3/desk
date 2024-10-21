
import os
from typing import Iterable

import fire

from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_clf_pred_save_path, get_comparison_save_path, get_label_path, get_csv_dataset_path


def enum_label_joined(dataset, run_name) -> Iterable[dict]:
    payload = read_csv(get_csv_dataset_path(dataset))
    preds = read_csv(get_clf_pred_save_path(run_name, dataset))
    labels = read_csv(get_label_path(dataset))

    assert len(payload) == len(preds)
    assert len(labels) == len(payload)

    for (id0, pred, _score), (id1, text), (id2, label) in zip(preds, payload, labels):
        assert id0 == id1
        assert id1 == id2
        yield {"id": id0, "prediction": int(pred), "label": int(label), "text": text}


def enum_wrong(dataset, run_name) -> Iterable[dict]:
    for e in enum_label_joined(dataset, run_name):
        if e["prediction"] != e["label"]:
            yield e


def enum_FP(dataset, run_name) -> Iterable[dict]:
    for e in enum_label_joined(dataset, run_name):
        if e["prediction"] == 1 and e["label"] == 0:
            yield e


def enum_FN(dataset, run_name) -> Iterable[dict]:
    for e in enum_label_joined(dataset, run_name):
        if e["prediction"] == 0 and e["label"] == 1:
            yield e