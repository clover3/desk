import fire

from desk_util.clf_util import clf_predict_w_predict_fn
from toxicity.hf_classifiers.get_clf import get_hf_classifier
from desk_util.runnable.run_eval_clf import run_eval_clf


def main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics=""
) -> None:
    predict_fn = get_hf_classifier(run_name)
    clf_predict_w_predict_fn(dataset, run_name, predict_fn)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(main)
