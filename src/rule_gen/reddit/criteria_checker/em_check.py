import numpy as np
import fire
from sklearn.metrics import confusion_matrix

from chair.list_lib import right
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.criteria_checker.feature_valuation import feature_valuation


def main(dataset, feature: str =""):
    def extract_feature(text):
        x_i = feature.lower() in text.lower()
        return x_i

    payload = load_csv_dataset_by_name(dataset)
    arr = []
    for data_id, text in payload:
        x_i = extract_feature(text)
        arr.append(x_i)
    arr = np.array(arr)

    def get_value(y_true, y_pred):
        m = confusion_matrix(y_true, y_pred)
        fp, tp = m[:, 1]
        return tp / (tp + fp), tp

    labels = load_labels(dataset)
    y = right(labels)
    v = get_value(y, arr)
    print("Value", v)


if __name__ == "__main__":
    fire.Fire(main)
