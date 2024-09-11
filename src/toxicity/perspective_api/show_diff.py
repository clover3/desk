import csv
from typing import Dict, Any
from datasets import load_dataset

from newbie.io_helper import load_predictions
from newbie.path_helper import get_dataset_pred_save_path, get_open_ai_mod_perspective_api_res_path
from newbie.read_perspective_labels import load_perspective_results


def convert_predictions_to_binary(predictions: Dict[str, str], target_string: str,
                                  case_sensitive: bool = False) -> Dict[str, int]:
    binary_predictions = {}
    for id, prediction in predictions.items():
        if not case_sensitive:
            prediction = prediction.lower()
            target_string = target_string.lower()
        binary_predictions[id] = 1 if target_string in prediction else 0
    return binary_predictions


def load_openai_moderation_predictions(run_name: str) -> Dict[str, str]:
    dataset_name = "openai-moderation"
    file_path = get_dataset_pred_save_path(run_name, dataset_name)
    ids, predictions = load_predictions(file_path)
    return dict(zip(ids, predictions))


def compare_predictions_and_save_csv(llama_guard_preds: Dict[str, int],
                                     perspective_preds: Dict[str, int],
                                     openai_preds: Dict[str, Dict[str, Any]],
                                     texts: Dict[str, str],
                                     output_file: str):
    all_ids = set(llama_guard_preds.keys()) & set(perspective_preds.keys()) & set(openai_preds.keys()) & set(
        texts.keys())
    n_items = len(all_ids)

    differing_instances = []
    for i, id in enumerate(all_ids):
        llama_pred = llama_guard_preds.get(id, 'N/A')
        perspective_pred = perspective_preds.get(id, 'N/A')
        openai_pred = openai_preds.get(id, {})
        text = texts.get(id, 'N/A')

        if llama_pred != perspective_pred:
            differing_instances.append((id, llama_pred, perspective_pred, openai_pred, text))

    print(f"\nInstances where predictions differ:")
    print(f"{'ID':<10} {'LlamaGuard':<15} {'Perspective API':<15} {'OpenAI':<15} {'Prompt Text':<50}")
    print("-" * 105)
    for id, llama_pred, perspective_pred, openai_pred, text in differing_instances:
        truncated_text = text[:47] + "..." if len(text) > 50 else text
        openai_pred_str = " ".join([k for k, v in openai_pred.items() if v])
        print(f"{id:<10} {llama_pred:<15} {perspective_pred:<15} {openai_pred_str:<15} {truncated_text:<50}")

    print(f"\nTotal differing instances: {len(differing_instances)} out of {n_items}")

    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['ID', 'LlamaGuard', 'Perspective API', 'OpenAI', 'Full Text'])
        for id, llama_pred, perspective_pred, openai_pred, text in differing_instances:
            openai_pred_str = " ".join([k for k, v in openai_pred.items() if v])
            csvwriter.writerow([id, llama_pred, perspective_pred, openai_pred_str, text])

    print(f"\nComparison results saved to {output_file}")


if __name__ == "__main__":
    run_name = "llama_guard2_prompt"

    # Load LlamaGuard predictions
    llama_guard_preds_s: dict[str, str] = load_openai_moderation_predictions(run_name)
    llama_guard_preds: dict[str, int] = convert_predictions_to_binary(llama_guard_preds_s, "S1")  # S9: Hate
    print(f"LlamaGuard Run: {run_name}")
    print(f"{sum(llama_guard_preds.values())} true out of {len(llama_guard_preds)}")

    # Load Perspective API predictions
    pers_labels = load_perspective_results("TOXICITY", 0.5, get_open_ai_mod_perspective_api_res_path())
    perspective_preds = {str(i): label for i, (text, label) in enumerate(pers_labels)}
    texts = {str(i): text for i, (text, _) in enumerate(pers_labels)}
    print("Perspective API")
    print(f"{sum(perspective_preds.values())} true out of {len(perspective_preds)}")

    # Load OpenAI dataset
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    openai_preds = {str(i): {k: v for k, v in item.items() if k != "prompt"} for i, item in enumerate(openai_dataset)}

    # Compare predictions and save to CSV
    output_file = f"{run_name}_comparison_results.csv"
    compare_predictions_and_save_csv(llama_guard_preds, perspective_preds, openai_preds, texts, output_file)