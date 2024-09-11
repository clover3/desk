import tqdm
import os

import fire
from tqdm import tqdm

from toxicity.cpath import output_root_path
from toxicity.io_helper import save_csv, read_csv_column, read_csv
from toxicity.path_helper import get_dataset_pred_save_path
from toxicity.predictors.get_predictor import get_llama_guard_like_predictor
from toxicity.runnable.run_eval import run_eval


def predict_inner(dataset, run_name):
    predict_fn = get_llama_guard_like_predictor(run_name)
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    payload: list[tuple[str, str]] = list(payload)

    def predict(e):
        id, text = e
        text, score = predict_fn([text])
        return id, text, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path: str = get_dataset_pred_save_path(run_name, dataset)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def main(
        run_name: str,
        dataset: str,
        do_eval=False,
        target_string="toxic",
        do_report=False,
        print_metrics=""
) -> None:
    predict_inner(dataset, run_name)
    if do_eval:
        run_eval(run_name, dataset, target_string,
                 do_report, print_metrics)



if __name__ == "__main__":
    fire.Fire(main)
