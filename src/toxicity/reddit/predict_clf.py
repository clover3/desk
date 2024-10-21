import fire

from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.classifier_loader.load_by_name import get_classifier
from toxicity.runnable.run_eval_clf import run_eval_clf


def predict_clf_main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics=""
) -> None:
    predict_fn = get_classifier(run_name)
    clf_predict_w_predict_fn(dataset, run_name, predict_fn)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
