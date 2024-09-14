from collections import Counter

from datasets import load_dataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

from chair.misc_lib import get_first, group_by
from chair.tab_print import tab_print_dict


def parse_selection(e, prefix):
    sel_idx = None
    for i in range(1, 6):
        sel = e[f"{prefix}.{i}"]
        if sel:
            if sel_idx is not None:
                raise ValueError()
            sel_idx = i
    return sel_idx


def main():
    all_data = load_dataset("toxigen/toxigen-data", name="annotations")["train"]

    all_entries = []
    for e in all_data:
        ai_score = parse_selection(e, "Answer.toAI")
        per_score = parse_selection(e, "Answer.toPER")
        final_score = max(ai_score, per_score)
        key = e["Input.text"]
        parsed_entry = key, final_score
        all_entries.append(parsed_entry)

    grouped = group_by(all_entries, get_first)
    print(len(grouped), len(all_entries))

    gold_idx = 0
    pred_idx = 1
    gold_labels = []
    predictions = []
    num_annot = Counter()
    for key, entries in grouped.items():
        if len(entries) > 1:
            gold_score = entries[gold_idx][1] > 3
            pred_score = entries[pred_idx][1] > 3
            gold_labels.append(gold_score)
            predictions.append(pred_score)
        num_annot[len(entries)] += 1

    print(num_annot)

    y_true = gold_labels
    y_pred = predictions
    metric = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "n": len(y_true)
    }
    tab_print_dict(metric)


"""
    9339 27450
    Counter({3: 8964, 1: 203, 2: 170, 6: 1, 9: 1})
    accuracy	0.8374562171628721
    precision	0.7474012474012474
    recall	0.7404737384140062
    f1	0.743921365752716
    confusion_matrix	[[5494, 729], [756, 2157]]
    n	9136
"""

if __name__ == "__main__":
    main()