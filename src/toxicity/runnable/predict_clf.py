import os

import fire
import tqdm
from tqdm import tqdm

from toxicity.cpath import output_root_path
from toxicity.hf_classifiers.get_clf import get_hf_classifier
from toxicity.io_helper import save_csv, read_csv
from toxicity.path_helper import get_clf_pred_save_path
from toxicity.runnable.run_eval_clf import run_eval_clf


def clf_predict_inner(dataset, run_name):
    predict_fn = get_hf_classifier(run_name)
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    def predict(e):
        id, text = e
        label, score = predict_fn(text)
        return id, label, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path = get_clf_pred_save_path(run_name, dataset)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics=""
) -> None:
    clf_predict_inner(dataset, run_name)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(main)
