import os
import fire

from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.cpath import output_root_path
from toxicity.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
from toxicity.runnable.run_eval_clf import run_eval_clf


def load_groups1():
    save_path = os.path.join(output_root_path, "reddit", "group", "group1.txt")
    groups = [l.strip() for l in open(save_path, "r").readlines()]
    return groups


def predict_clf_main(
        run_name: str,
        do_eval=True,
        do_report=True,
        print_metrics=""
) -> None:
    predict_fn = get_classifier_pipeline(run_name)

    group = load_groups1()
    for target in group:
        dataset = f"{target}_val_1K"
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
