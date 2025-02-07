import numpy as np

from chair.list_lib import right
from desk_util.path_helper import load_clf_pred
from desk_util.runnable.run_eval import load_labels


def load_prediction_as_features(run_name_iter, dataset):
    xs_list = []
    for run_name in run_name_iter:
        preds = load_clf_pred(dataset, run_name)
        xs = [e[1] for e in preds]
        xs_list.append(xs)
    per_feature = np.array(xs_list)
    X = np.transpose(per_feature, [1, 0])
    return X


def load_dataset_from_predictions(run_name_iter, dataset):
    X = load_prediction_as_features(run_name_iter, dataset)
    labels = load_labels(dataset)
    y = right(labels)
    return X, y


