
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from toxicity.clf_util import eval_prec_recall_f1_acc, print_evaluation_results
from toxicity.llama_guard.output_convertor import convert_predictions_to_binary
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import load_predictions
from toxicity.path_helper import get_dataset_pred_save_path

# Example usage
if __name__ == "__main__":
    # Example predictions
    run_name = "llama_guard2_prompt"
    dataset_name: str = 'toxigen_train_head'


    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    ids, text_predictions = load_predictions(save_path)

    # Convert to binary predictions
    target_string = "S1"
    predictions: list[int] = convert_predictions_to_binary(text_predictions, target_string)

    print(f"LlamaGuard Run: {run_name}")
    print(f"{sum(predictions)} true out of {len(predictions)}")
    test_dataset: ToxigenBinary = ToxigenBinary("test")

    labels = [e['label'] for e in test_dataset]
    print("Toxigen Human Annotations")
    print(f"{sum(labels)} true out of {len(labels)}")

    performance_metrics = eval_prec_recall_f1_acc(labels, predictions)
    print("\nEvaluation:")
    print_evaluation_results(performance_metrics)
