import sys
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator
from datasets import load_dataset
from toxicity.clf_util import eval_prec_recall_f1_acc, print_evaluation_results
from toxicity.llama_guard.output_convertor import convert_predictions_to_binary
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import load_predictions, read_csv
from toxicity.path_helper import get_dataset_pred_save_path


def subset_by_range():
    # Example predictions
    run_name = "llama_guard2_prompt"
    split = "test"
    dataset_name: str = f'toxigen_{split}'
    print(f"split={split}")
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    # Convert to binary predictions
    target_string = "S1"
    all_preds: list[int] = convert_predictions_to_binary(text_predictions, target_string)

    todo = [
        (0, 300),
        (300, 600),
        (600, 900),
        (900, 1200)
    ]
    test_dataset: ToxigenBinary = ToxigenBinary("test")
    all_labels = [e['label'] for e in test_dataset]

    for st, ed in todo:
        preds = all_preds[st:ed]
        labels = all_labels[st:ed]

        print(f"{st}:{ed}: Preds {sum(preds)} true out of {len(preds)}")
        print(f"{st}:{ed}: labels {sum(labels)} true out of {len(labels)}")
        performance_metrics = eval_prec_recall_f1_acc(labels, preds)
        print_evaluation_results(performance_metrics)


def subset_by_source():
    # Example predictions
    run_name = "llama_guard2_prompt"
    split = "test"
    dataset_name: str = f'toxigen_{split}'
    print(f"split={split}")
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    # Convert to binary predictions
    target_string = "S1"
    all_preds: list[int] = convert_predictions_to_binary(text_predictions, target_string)

    test_dataset: ToxigenBinary = ToxigenBinary(split)
    all_labels = [e['label'] for e in test_dataset]
    dataset = load_dataset("toxigen/toxigen-data", name="annotated")[split]

    todo = ["human","cbs", "topk"]
    for target_source in todo:
        indices = [i for i, e in enumerate(test_dataset) if e["actual_method"] == target_source]
        preds = [all_preds[i] for i in indices]
        labels = [all_labels[i] for i in indices]

        print(f"Preds {sum(preds)} true out of {len(preds)}")
        print(f"labels {sum(labels)} true out of {len(labels)}")
        performance_metrics = eval_prec_recall_f1_acc(labels, preds)
        print_evaluation_results(performance_metrics)


# Example usage
if __name__ == "__main__":
    subset_by_source()