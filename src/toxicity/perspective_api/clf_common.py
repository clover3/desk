import csv
from typing import Tuple, List, Dict

from sklearn.metrics import classification_report
from tqdm import tqdm
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from classifiers import TextGenerator
from clf_util import BinaryDataset


def run_predictions(model: TextGenerator, dataset: BinaryDataset) -> Iterator[tuple[str, str]]:
    progress_bar = tqdm(total=len(dataset), desc="Predicting")

    for i in range(len(dataset)):
        item: Dict[str, str] = dataset[i]
        id: str = item['id']
        text: str = item['text']

        prediction: str = model.predict([text])[0]  # predict expects a list, but we're processing one item at a time
        yield id, prediction
        progress_bar.update(1)

    progress_bar.close()



def print_performance_report(test_labels: List[int], test_preds: List[str]) -> None:
    print('\nTest Performance:')
    print(classification_report(test_labels, test_preds, target_names=['safe', 'unsafe']))


def save_results_w_label(ids: List[str], true_labels: List[int], predictions: List[str], file_path: str) -> None:
    with open(file_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['ID', 'True Label', 'Predicted Label'])
        for id, true_label, pred in zip(ids, true_labels, predictions):
            writer.writerow([id, true_label, pred])
    print(f"Results saved to {file_path}")
