from typing import List, Tuple

import fire

from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.clf_util import eval_prec_recall_f1_acc
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_dataset_pred_save_path, get_label_path, get_clf_pred_save_path


def align_preds_and_labels(
        preds: List[Tuple[str, int, float]],
        labels: List[Tuple[str, int]]) -> Tuple[List[int], List[int], List[float]]:
    pred_dict = {p[0]: (p[1], p[2]) for p in preds}  # data_id: (binary_pred, raw_score)
    label_dict = {l[0]: l[1] for l in labels}  # data_id: label

    aligned_preds = []
    aligned_labels = []
    aligned_scores = []

    for data_id in label_dict:
        if data_id in pred_dict:
            aligned_preds.append(pred_dict[data_id][0])
            aligned_labels.append(label_dict[data_id])
            aligned_scores.append(pred_dict[data_id][1])

    return aligned_preds, aligned_labels, aligned_scores


def clf_eval(
        preds: List[Tuple[str, int, float]],
        labels: List[Tuple[str, int]]):
    from sklearn.metrics import auc, roc_curve

    aligned_preds, aligned_labels, aligned_scores = align_preds_and_labels(preds, labels)
    if not aligned_preds:
        raise ValueError("No matching data_ids found between predictions and labels")
    score_d = eval_prec_recall_f1_acc(aligned_labels, aligned_preds)
    fpr, tpr, thresholds = roc_curve(aligned_labels, aligned_scores)
    auc_val = auc(fpr, tpr)
    score_d["auc"] = auc_val
    return score_d


def load_predictions(pred_path, target_string="S1") -> list[tuple[str, int, float]]:
    preds = read_csv(pred_path)
    target_string = target_string.lower()
    output: list[tuple[str, int, float]] = []
    for data_id, gen_text, score_s in preds:
        text = gen_text.lower()
        pred = 1 if target_string in text else 0

        score = float(score_s)
        score = score if pred else -score
        output.append((data_id, pred, score))
    return output


def load_labels(dataset) -> list[tuple[str, int]]:
    rows = read_csv(get_label_path(dataset))
    return [(data_id, int(label)) for data_id, label in rows]


def run_eval_from_gen_out(run_name, dataset, target_string,
                          do_report=False,
                          print_metrics=""):
    save_path: str = get_dataset_pred_save_path(run_name, dataset)
    preds = load_predictions(save_path, target_string)
    labels = load_labels(dataset)
    score_d = clf_eval(preds, labels)
    metrics_to_report = ["accuracy", "f1"]
    if print_metrics:
        if isinstance(print_metrics, str):
            print_metrics = [print_metrics]
    else:
        print_metrics = ["accuracy", "f1", "auc", "n"]

    for metric in print_metrics:
        print(f"{metric}\t{score_d[metric]}")

    if len(preds) != score_d['n']:
        msg = f"Evaluated on {score_d['n']} samples"
        msg += f" while prediction has {len(preds)} samples"
        print(msg)

    if do_report:
        proxy = get_task_manager_proxy()
        for metric in metrics_to_report:
            metric_short = metric[:3]
            proxy.report_number(run_name, score_d[metric], dataset, metric_short)


if __name__ == "__main__":
    fire.Fire(run_eval_from_gen_out)
