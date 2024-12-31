import os

import fire

from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_clf_pred_save_path, get_comparison_save_path


def do_reddit_pred_compare(dataset, run1, run2):
    def load(run_name):
        save_path = get_clf_pred_save_path(run_name, dataset)
        return read_csv(save_path)

    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    d1 = load(run1)
    d2 = load(run2)
    assert len(d1) == len(d2)
    assert len(d1) == len(payload)

    def iter():
        for (id0, text), (id1, pred1, score1), (id2, pred2, score2) in zip(payload, d1, d2):
            assert id0 == id1
            assert id1 == id2
            if pred1 != pred2:
                yield id1, pred1, pred2, text

    run_name = f"{run1}_{run2}"
    s = get_comparison_save_path(run_name, dataset)
    save_csv(iter(), s)


if __name__ == "__main__":
    fire.Fire(do_reddit_pred_compare)
