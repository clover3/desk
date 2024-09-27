from typing import List, Tuple

import fire
from toxicity.io_helper import save_csv
from toxicity.path_helper import get_dataset_pred_save_path, get_wrong_pred_save_path
from toxicity.runnable.run_eval import load_predictions, load_labels


def align_preds_and_labels(
        preds: List[Tuple[str, int, float]],
        labels: List[Tuple[str, int]]) -> List[Tuple[str, int, int]]:
    pred_dict = {p[0]: p[1] for p in preds}  # data_id: binary_pred
    label_dict = {l[0]: l[1] for l in labels}  # data_id: label

    aligned_data = []
    for data_id in label_dict:
        if data_id in pred_dict:
            aligned_data.append((data_id, pred_dict[data_id], label_dict[data_id]))

    return aligned_data


def save_incorrect_predictions(aligned_data: List[Tuple[str, int, int]], output_path: str):
    incorrect_predictions = [(data_id, pred, label) for data_id, pred, label in aligned_data if pred != label]
    save_csv(incorrect_predictions, output_path)
    print(f"Incorrect predictions saved to: {output_path}")
    print(f"Number of incorrect predictions: {len(incorrect_predictions)}")


def main(run_name, dataset, target_string="S1"):
    save_path: str = get_dataset_pred_save_path(run_name, dataset)
    preds = load_predictions(save_path, target_string)
    labels = load_labels(dataset)
    aligned_data = align_preds_and_labels(preds, labels)

    save_path: str = get_wrong_pred_save_path(run_name, dataset)
    save_incorrect_predictions(aligned_data, save_path)


if __name__ == "__main__":
    fire.Fire(main)
