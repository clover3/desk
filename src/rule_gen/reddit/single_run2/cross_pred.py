import os

import fire

from desk_util.clf_util import clf_predict_w_predict_fn
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.cpath import output_root_path
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier


def load_groups1():
    save_path = os.path.join(output_root_path, "reddit", "group", "group1.txt")
    groups = [l.strip() for l in open(save_path, "r").readlines()]
    return groups


def predict_clf_main(
        run_name: str,
        do_eval=False,
        dataset_fmt="{}_2_val_100",
        do_report=False,
        print_metrics="",
        overwrite=False,
) -> None:
    predict_fn = get_classifier(run_name)

    group = load_groups1()
    for target in group:
        dataset = dataset_fmt.format(target)
        clf_predict_w_predict_fn(dataset, run_name, predict_fn, overwrite)

        if do_eval:
            run_eval_clf(run_name, dataset,
                         do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
