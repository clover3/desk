import csv
from typing import List

import fire

from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_dataset_pred_save_path, get_comparison_save_path


def compare_predictions_and_save_csv(
        preds: List[int],
        labels: List[int],
        texts: List[str],
        output_file: str,
        print_only_diff
):
    assert len(preds) == len(labels) == len(texts), "All input lists must have the same length"
    n_items = len(preds)
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ID', 'Pred', 'Label', 'Text'])
        for i in range(n_items):
            pred = preds[i]
            label = labels[i]
            text = texts[i]
            do_print = not print_only_diff or (pred != label)
            if do_print:
                csvwriter.writerow([str(i), pred, label, text])

    print(f"\nComparison results saved to {output_file}")


def main(run_name, split, target_string="S1", n_pred=None, print_only_diff=True):
    if n_pred is None:
        dataset_name: str = f'toxigen_{split}'
    else:
        dataset_name: str = f'toxigen_{split}_head_{n_pred}'

    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    text_predictions = [e[1] for e in read_csv(save_path)]
    preds: List[int] = [1 if target_string in pred else 0 for pred in text_predictions]
    print(f"Run: {run_name}")
    print(f"{sum(preds)} true out of {len(preds)}")
    test_dataset: ToxigenBinary = ToxigenBinary(split)
    texts = [e['text'] for e in test_dataset][:n_pred]
    lables = [e['label'] for e in test_dataset][:n_pred]
    print("Human Annotations")
    print(f"{sum(lables)} true out of {len(lables)}")
    output_file = get_comparison_save_path(run_name, dataset_name)
    compare_predictions_and_save_csv(
        preds, lables, texts, output_file, print_only_diff)


# Example usage
if __name__ == "__main__":
    fire.Fire(main)
