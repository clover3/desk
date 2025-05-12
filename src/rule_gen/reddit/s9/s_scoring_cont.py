import logging

import fire

from desk_util.llm_clf_util import llm_predict_w_predict_fn
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.display.show_avg_p_r_f1 import show_avg_p_r_f1
from rule_gen.reddit.llm_inf.get_predictor import get_llm_predictor
from rule_gen.reddit.path_helper import get_split_subreddit_list

LOG = logging.getLogger(__name__)


def predict_sb_split(
        run_name_fmt: str,
        split: str,
        do_eval=False,
        overwrite=False,
) -> None:
    sb_list = get_split_subreddit_list(split)
    dataset_fmt = "{}_2_train_100",

    st = 100

    for sb in sb_list:
        run_name = run_name_fmt.format(sb)
        dataset = dataset_fmt.format(sb)
        llm_pred_fn = get_llm_predictor(run_name)
        llm_predict_w_predict_fn(dataset, run_name, llm_pred_fn, overwrite)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         True)

    show_avg_p_r_f1(dataset_fmt, run_name_fmt.format, split)


if __name__ == "__main__":
    fire.Fire(predict_sb_split)
