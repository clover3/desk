import fire

from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.cpath import output_root_path
import os


def load_groups1():
    save_path = os.path.join(output_root_path, "reddit", "group", "group1.txt")
    groups = [l.strip() for l in open(save_path, "r").readlines()]
    return groups


def predict_clf_main(
        run_name: str,
        do_eval=False,
        dataset_fmt="{}_2_val_100",
        do_report=True,
        print_metrics="",
) -> None:
    group = load_groups1()
    for target in group:
        dataset = dataset_fmt.format(target)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
