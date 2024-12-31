import fire

from taskman_client.task_proxy import get_task_manager_proxy
from desk_util.path_helper import load_clf_pred
from desk_util.runnable.run_eval import load_labels, clf_eval


def run_eval_clf(run_name,
                 dataset,
                 do_report=False,
                 print_metrics=""):
    preds = load_clf_pred(dataset, run_name)
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
