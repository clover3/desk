import fire

from desk_util.clf_util import eval_prec_recall_f1_acc
from chair.list_lib import right, left
from toxicity.llama_guard.output_convertor import parse_predictions
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_dataset_pred_save_path
from sklearn.metrics import auc, roc_curve
from taskman_client.task_proxy import get_task_manager_proxy


def get_dataset_split(split):
    if "train_fail" in split or "1000_para" == split:
        return split
    if split == "head_100_para_clean":
        return "train"

    if "train" in split:
        ds_split = "train"
    elif "test" in split:
        ds_split = "test"
    else:
        raise ValueError(f"Unknown split {split}")
    return ds_split


def load_labels(dataset):
    ds_split = get_dataset_split(dataset)
    test_dataset: ToxigenBinary = ToxigenBinary(ds_split)
    labels = [e['label'] for e in test_dataset]
    if dataset == "head_100_para_clean":
        return labels[:100]
    return labels


# Problem: Loading label is dirty

def run_toxigen_eval(run_name, split, do_report=False, n_pred=None, target_string=None):
    if n_pred is None:
        dataset_name: str = f'toxigen_{split}'
    else:
        dataset_name: str = f'toxigen_{split}_head_{n_pred}'

    print(f"dataset_name={dataset_name}")
    print(f"split={split}")
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    raw_scores = [float(e[2]) for e in preds]

    if target_string is None:
        target_string = "S1"
    parsed: list[tuple[int, float]] = parse_predictions(text_predictions, raw_scores, target_string)
    predictions = left(parsed)
    scores = right(parsed)
    n_item = len(parsed)
    labels = load_labels(split)
    if n_item < len(labels):
        print(f"Run {run_name} has {n_item} items. Adjust labels from {len(labels)} to {n_item}")
    labels = labels[:n_item]
    print("Toxigen Human Annotations")
    print(f"{sum(labels)} true out of {len(labels)}")
    performance_metrics = eval_prec_recall_f1_acc(labels, predictions)
    acc = performance_metrics["accuracy"]
    f1 = performance_metrics["f1"]
    fpr, tpr, thresholds = roc_curve(labels, scores)
    auc_val = auc(fpr, tpr)
    print("auc", auc_val)
    print("f1", f1)
    print("acc", acc)

    if do_report:
        proxy = get_task_manager_proxy()
        proxy.report_number(run_name, auc_val, dataset_name, "auc")
        proxy.report_number(run_name, f1, dataset_name, "f1")
        proxy.report_number(run_name, acc, dataset_name, "acc")


if __name__ == "__main__":
    fire.Fire(run_toxigen_eval)
