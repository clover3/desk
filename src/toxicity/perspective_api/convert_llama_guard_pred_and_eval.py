from typing import List, Tuple

from newbie.clf_util import convert_predictions_to_binary, evaluate_performance, print_evaluation_results
from newbie.io_helper import load_predictions
from newbie.list_lib import right
from newbie.path_helper import get_dataset_pred_save_path, get_open_ai_mod_perspective_api_res_path
from newbie.read_perspective_labels import load_perspective_results
"""
Loaded 1680 predictions from /sfs/qumulo/qhome/qdb5sn/work/GRACE/outputs/openai-moderation/llama_guard2_prompt.csv
LlamaGuard Run: llama_guard2_prompt
255 true out of 1680
Perspective API
347 true out of 1680

LlamaGuard Performance Evaluation:
Accuracy: 0.8048
Precision: 0.5373
Recall: 0.3948
F1 Score: 0.4551
Confusion Matrix:
[[1215  118]
 [ 210  137]]
 
Loaded 1680 predictions from /sfs/qumulo/qhome/qdb5sn/work/GRACE/outputs/openai-moderation/llama_guard2.csv
LlamaGuard Run: llama_guard2
132 true out of 1680
Perspective API
347 true out of 1680

LlamaGuard Performance Evaluation:
Accuracy: 0.7970
Precision: 0.5227
Recall: 0.1988
F1 Score: 0.2881
Confusion Matrix:
[[1270   63]
 [ 278   69]]

"""


def load_openai_moderation_predictions(run_name: str) -> Tuple[List[str], List[str]]:
    dataset_name = "openai-moderation"
    file_path = get_dataset_pred_save_path(run_name, dataset_name)
    ids, predictions = load_predictions(file_path)
    return ids, predictions


# Example usage
if __name__ == "__main__":
    # Example predictions
    run_name = "llama_guard2"
    ids, text_predictions = load_openai_moderation_predictions(run_name)

    # Convert to binary predictions
    target_string = "S1"
    target_string = "S9"  # S9: Hate
    llama_guard_predictions = convert_predictions_to_binary(text_predictions, target_string)

    print(f"LlamaGuard Run: {run_name}")
    print(f"{sum(llama_guard_predictions)} true out of {len(llama_guard_predictions)}")

    # Load Perspective API results and generate IDs
    pers_labels: List[Tuple[str, int]] = load_perspective_results("TOXICITY", 0.5,
                                                                  get_open_ai_mod_perspective_api_res_path())
    perspective_ids = [str(i) for i in range(len(pers_labels))]
    perspective_labels = right(pers_labels)

    print("Perspective API")
    print(f"{sum(perspective_labels)} true out of {len(perspective_labels)}")

    # Ensure that we're comparing the same samples
    common_ids = set(ids) & set(perspective_ids)
    if len(common_ids) != len(ids) or len(common_ids) != len(perspective_ids):
        print(f"Warning: Mismatch in sample sizes. Common samples: {len(common_ids)}")

    # Filter predictions and labels to include only common IDs
    llama_guard_filtered = [pred for id, pred in zip(ids, llama_guard_predictions) if id in common_ids]
    perspective_filtered = [label for id, label in zip(perspective_ids, perspective_labels) if id in common_ids]

    # Evaluate LlamaGuard performance
    performance_metrics = evaluate_performance(perspective_filtered, llama_guard_filtered)

    print("\nLlamaGuard Performance Evaluation:")
    print_evaluation_results(performance_metrics)
