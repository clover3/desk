import os

import fire

from chair.tab_print import print_table
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_clf_pred_save_path, get_comparison_save_path
from desk_util.runnable.run_eval import load_labels


def do_reddit_pred_compare(dataset, run_name, stdout=False):
    def load(run_name):
        save_path = get_clf_pred_save_path(run_name, dataset)
        return read_csv(save_path)

    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)
    payload_d = dict(payload)
    labels = load_labels(dataset)
    labels_d = dict(labels)

    preds = load(run_name)
    output = []
    for data_id, pred, score in preds:
        l = labels_d[data_id]
        t = payload_d[data_id]
        if int(l) != int(pred):
            output.append((data_id, l, pred, t))


    run_name = f"{run_name}"
    if stdout:
        print_table(output)
    else:
        s = get_comparison_save_path(run_name, dataset)
        save_csv(output, s)
        print("Saved at {}".format(s))


if __name__ == "__main__":
    fire.Fire(do_reddit_pred_compare)
