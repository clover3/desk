import logging

import fire

from desk_util.llm_clf_util import llm_predict_w_predict_fn
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.display.show_avg_p_r_f1 import show_avg_p_r_f1
from rule_gen.reddit.llama.token_scoring import get_predictor_from_run_name
from rule_gen.reddit.path_helper import get_split_subreddit_list

LOG = logging.getLogger(__name__)


def predict_sb_split(
        run_name_fmt: str,
        split: str,
        dataset_fmt = "{}_2_val_100",
        do_eval=False,
        overwrite=False,
) -> None:
    todo = get_split_subreddit_list(split)

    for sb in todo:
        run_name = run_name_fmt.format(sb)
        dataset = dataset_fmt.format(sb)
        llm_pred_fn = get_predictor_from_run_name(run_name)
        llm_predict_w_predict_fn(dataset, run_name, llm_pred_fn, overwrite)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         True)

    show_avg_p_r_f1(dataset_fmt, run_name_fmt.format, split)


if __name__ == "__main__":
    fire.Fire(predict_sb_split)
