import fire

from chair.list_lib import left
from toxicity.llama_guard.output_convertor import parse_predictions
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_dataset_pred_save_path, get_toxigen_failure_save_path
from toxicity.toxigen_eval_analysis.run_eval import get_dataset_split


def run_toxigen_eval(run_name, split, n_pred=None):
    ds_split = get_dataset_split(split)
    if n_pred is None:
        dataset_name: str = f'toxigen_{split}'
    else:
        dataset_name: str = f'toxigen_{split}_head_{n_pred}'

    print(f"dataset_name={dataset_name}")
    print(f"split={split}")
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    raw_scores = [float(e[2]) for e in preds]
    # Convert to binary predictions
    target_string = "S1"
    parsed: list[tuple[int, float]] = parse_predictions(text_predictions, raw_scores, target_string)
    predictions = left(parsed)
    n_item = len(parsed)
    test_dataset: ToxigenBinary = ToxigenBinary(ds_split)
    labels = [e['label'] for e in test_dataset]
    if n_item < len(labels):
        print(f"Run {run_name} has {n_item} items. Adjust labels from {len(labels)} to {n_item}")
    labels = labels[:n_item]
    print("Toxigen Human Annotations")
    print(f"{sum(labels)} true out of {len(labels)}")
    rows = []
    for i in range(n_item):
        if labels[i] != predictions[i]:
            rows.append((i, test_dataset[i]["text"], labels[i]))

    save_path = get_toxigen_failure_save_path(dataset_name, run_name)
    save_csv(rows, save_path)


if __name__ == "__main__":
    fire.Fire(run_toxigen_eval)
