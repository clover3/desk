from typing import List

import fire

from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_dataset_pred_save_path, get_comparison_save_path

import matplotlib.pyplot as plt
import numpy as np


def visualize_predictions(predictions):
    # Convert TRUE/FALSE to 1/0
    data = np.array([1 if pred else 0 for pred in predictions])

    # Create a figure and axis
    fig, ax = plt.subplots(figsize=(12, 2))

    # Create the heatmap
    heatmap = ax.imshow(data.reshape(1, -1), cmap='RdYlGn', aspect='auto', interpolation='nearest')

    # Remove y-axis ticks
    ax.set_yticks([])

    # Set x-axis ticks at multiples of 5
    num_predictions = len(predictions)
    xticks = np.arange(4, num_predictions, 5)
    ax.set_xticks(xticks)
    ax.set_xticklabels((xticks + 1).astype(int))

    # Add minor ticks for all other positions
    ax.set_xticks(np.arange(-0.5, num_predictions, 1), minor=True)
    ax.tick_params(axis='x', which='minor', length=0)

    # Add gridlines at the minor ticks
    ax.grid(which='minor', axis='x', linestyle='-', linewidth=0.5, alpha=0.2)

    # Add labels and title
    ax.set_xlabel('Prediction Number')
    ax.set_title('Prediction Correctness Visualization')

    # Add a color bar
    cbar = plt.colorbar(heatmap, ax=ax, orientation='vertical', pad=0.02)
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(['FALSE', 'TRUE'])

    # Show the plot
    plt.tight_layout()
    plt.show()


# Example usage
predictions = [True, False, True, True, False, True, False, True, True, True] * 5  # 50 predictions
visualize_predictions(predictions)
def compare_predictions(
        preds: List[int],
        labels: List[int],
        texts: List[str],
        output_file: str,
):
    assert len(preds) == len(labels) == len(texts), "All input lists must have the same length"
    n_items = len(preds)
    all_res = []
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        for i in range(n_items):
            pred = preds[i]
            label = labels[i]
            text = texts[i]
            correct = pred == label
            all_res.append(correct)
            # do_print = not print_only_diff or (pred != label)
    return all_res



def main(run_name, split, target_string="S1", n_pred=None):
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
    correctness = compare_predictions(
        preds, lables, texts, output_file)
    print(correctness)
    visualize_predictions(correctness)



# Example usage
if __name__ == "__main__":
    fire.Fire(main)
