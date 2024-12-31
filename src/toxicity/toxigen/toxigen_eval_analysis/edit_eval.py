


# Compare "Before" run and "Edit run"
# In "Before" model,
#   TP, TN are considered locality instances
#   FP, FN are considered generalizability


from desk_util.clf_util import eval_prec_recall_f1_acc
from toxicity.llama_guard.output_convertor import convert_predictions_to_binary
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_dataset_pred_save_path


def per_item_eval(labels: list[int], predictions: list[int]) -> list[str]:
    """
    Evaluate binary (0/1) predictions against true labels.

    Args:
    labels (list[int]): List of true labels (0 or 1)
    predictions (list[int]): List of predicted labels (0 or 1)

    Returns:
    list[str]: List of evaluation results ("TP", "TN", "FP", or "FN" for each prediction)
    """
    if len(labels) != len(predictions):
        raise ValueError("Labels and predictions must have the same length")

    results = []
    for label, prediction in zip(labels, predictions):
        if label == 1 and prediction == 1:
            results.append("TP")
        elif label == 0 and prediction == 0:
            results.append("TN")
        elif label == 0 and prediction == 1:
            results.append("FP")
        elif label == 1 and prediction == 0:
            results.append("FN")
        else:
            raise ValueError(f"Invalid label or prediction: {label}, {prediction}")

    return results


# Example usage
if __name__ == "__main__":
    # Example predictions
    ds_split = "train"
    split = "head_1000"
    dataset_name: str = f'toxigen_train_{split}'
    print(dataset_name)
    
    base_run_name = "llama_guard2_prompt"
    edit_run_name = "lg2_3"

    def load_predictions_local(run_name):
        save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
        preds = read_csv(save_path)
        text_predictions = [e[1] for e in preds]
        # Convert to binary predictions
        target_string = "S1"
        predictions: list[int] = convert_predictions_to_binary(text_predictions, target_string)
        print(f"Run={run_name}: {sum(predictions)} true out of {len(predictions)}")
        return predictions

    base_predictions = load_predictions_local(base_run_name)
    edit_predictions = load_predictions_local(edit_run_name)
    
    n_item = len(base_predictions)
    assert len(edit_predictions) == n_item
    test_dataset: ToxigenBinary = ToxigenBinary(ds_split)

    labels = [e['label'] for e in test_dataset]
    labels = labels[:n_item]

    print("Toxigen Human Annotations")
    print(f"{sum(labels)} true out of {len(labels)}")

    base_res = per_item_eval(labels, base_predictions)
    edit_res = per_item_eval(labels, edit_predictions)
    locality_indices = [i for i in range(n_item) if base_res[i] in ["TP", "TN"]]
    general_indices = [i for i in range(n_item) if base_res[i] in ["FP", "FN"]]

    def subset_eval(all_preds, indices):
        subset_preds = [all_preds[i] for i in indices]
        subset_labels = [labels[i] for i in indices]
        performance_metrics = eval_prec_recall_f1_acc(subset_labels, subset_preds)
        print(performance_metrics)
        return performance_metrics["f1"]


    print("\nPost-edit Evaluation:")
    locality_score = subset_eval(edit_predictions, locality_indices)
    general_score = subset_eval(edit_predictions, general_indices)
    print("Locality", locality_score)
    print("General", general_score)

    # proxy = get_task_manager_proxy()
    # proxy.report_number(run_name, performance_metrics["f1"], dataset_name, "f1")
