import logging

import evaluate
import numpy as np

LOG = logging.getLogger(__name__)


def get_compute_metrics():
    clf_metrics = evaluate.combine(["accuracy", "f1", "precision", "recall"])

    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        predictions = np.argmax(logits, axis=-1)
        return clf_metrics.compute(predictions=predictions, references=labels)

    return compute_metrics
