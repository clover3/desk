from newbie.classifiers import LlamaGuard, TextGenerator
from newbie.io_helper import save_predictions
from newbie.datasets.open_ai_modetation_dataset import ModerationDatasetAsBinary
from typing import List

from newbie.path_helper import get_dataset_pred_save_path
from newbie.perspective_api.clf_common import run_predictions, print_performance_report


def main() -> None:
    dataset_name: str = 'openai-moderation'
    test_dataset: ModerationDatasetAsBinary = ModerationDatasetAsBinary()
    model: TextGenerator = LlamaGuard()
    run_name: str = "llama_guard2"

    # Run predictions
    test_preds, test_ids = run_predictions(model, test_dataset)
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)

    # Save predictions
    save_predictions(test_ids, test_preds, save_path)

    # Get test labels
    test_labels: List[int] = [test_dataset[int(id)]['label'] for id in test_ids]

    # Save results with labels
    print_performance_report(test_labels, test_preds)


def main_llama_guard2_prompt() -> None:
    dataset_name: str = 'openai-moderation'
    test_dataset: ModerationDatasetAsBinary = ModerationDatasetAsBinary()
    model: TextGenerator = LlamaGuard(use_toxicity=True)
    run_name: str = "llama_guard2_prompt"

    # Run predictions
    test_preds, test_ids = run_predictions(model, test_dataset)
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)

    # Save predictions
    save_predictions(test_ids, test_preds, save_path)


if __name__ == "__main__":
    main_llama_guard2_prompt()
