import logging

import fire

from desk_util.io_helper import save_csv
from desk_util.path_helper import get_clf_pred_save_path, load_clf_pred
from desk_util.runnable.run_eval_clf import run_eval_clf

LOG = logging.getLogger(__name__)


def combine_clf_pred_and(
        run_name1: str,
        run_name2: str,
        save_run_name: str,
        dataset: str,
        do_eval=False,
) -> None:
    preds1 = load_clf_pred(dataset, run_name1)
    preds2 = load_clf_pred(dataset, run_name2)
    pred_itr = []
    for e1, e2 in zip(preds1, preds2):
        data_id1, p1, s1 = e1
        data_id2, p2, s2 = e2
        assert data_id1 == data_id2
        new_p = int(p1 or p2)
        s_new = max(s1, s2)
        out_e = data_id1, new_p, s_new
        pred_itr.append(out_e)

    save_path = get_clf_pred_save_path(save_run_name, dataset)
    save_csv(pred_itr, save_path)

    if do_eval:
        run_eval_clf(save_run_name, dataset,
                     True)


if __name__ == "__main__":
    fire.Fire(combine_clf_pred_and)
