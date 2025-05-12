import fire
import logging
from desk_util.llm_clf_util import llm_predict_w_predict_fn
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.llm_inf.get_predictor import get_llm_predictor

LOG = logging.getLogger(__name__)


def predict_llm_main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics="",
        overwrite=False,
) -> None:
    llm_pred_fn = get_llm_predictor(run_name)
    llm_predict_w_predict_fn(dataset, run_name, llm_pred_fn, overwrite)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_llm_main)
