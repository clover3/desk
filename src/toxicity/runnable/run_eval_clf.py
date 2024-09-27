import fire

from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_clf_pred_save_path
from toxicity.runnable.run_eval import load_labels, clf_eval


def run_eval_clf(run_name,
                 dataset,
                 do_report=False,
                 print_metrics=""):
    save_path: str = get_clf_pred_save_path(run_name, dataset)
    raw_preds = read_csv(save_path)
    preds = [(data_id, int(pred), float(score)) for data_id, pred, score in raw_preds]
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
    fire.Fire(run_eval_clf)
