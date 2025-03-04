import numpy as np
from collections import Counter

from rule_gen.reddit.single_run2.mutual_info import confusion_to_xy


def entropy(y):
    if len(y) == 0:
        return 0.0

    counter = Counter(y)
    probs = np.array([count / len(y) for count in counter.values()])
    return -np.sum(probs * np.log2(probs + np.finfo(float).eps))


def rule_value(prev_y: list[int], feature_preds: list[int]) -> float:
    y = np.array(prev_y)
    before_entropy = entropy(prev_y)

    def entropy_per_label(feature_values):
        indices = feature_values == np.array(feature_preds)
        subset_labels = y[indices]
        prob = len(subset_labels) / len(y)
        subset_entropy = entropy(subset_labels)

        # Store split information
        return {
            'count': len(subset_labels),
            'entropy': subset_entropy,
            'prob': prob,
            'class_distribution': Counter(subset_labels)
        }

    weighted_entropy = 0
    for v in [0, 1]:
        ret = entropy_per_label(v)
        weighted_entropy += ret["prob"] * ret["entropy"]

    gain = before_entropy - weighted_entropy
    return gain


def information_gain(X, y, feature_idx=0):
    if not isinstance(y, np.ndarray):
        y = np.array(y)



    parent_entropy = entropy(y)
    feature_values = X[:, feature_idx]
    unique_values = np.unique(feature_values)

    # Calculate weighted entropy after split
    weighted_entropy = 0
    split_info = {}

    for value in unique_values:
        indices = feature_values == value
        subset_labels = y[indices]

        # Calculate probability of this value occurring
        prob = len(subset_labels) / len(y)

        # Calculate entropy for this subset
        subset_entropy = entropy(subset_labels)

        # Add weighted entropy to total
        weighted_entropy += prob * subset_entropy

        # Store split information
        split_info[value] = {
            'count': len(subset_labels),
            'entropy': subset_entropy,
            'class_distribution': Counter(subset_labels)
        }

    # Calculate information gain
    info_gain = parent_entropy - weighted_entropy

    return info_gain, {
        'parent_entropy': parent_entropy,
        'weighted_entropy': weighted_entropy,
        'split_info': split_info
    }


# Example usage
if __name__ == "__main__":
    # Create sample dataset
    tp, fp, fn, tn = 19, 4, 25, 40
    X, y = confusion_to_xy(tp, fp, fn, tn)

    # Calculate information gain for weather feature (index 1)
    gain, details = information_gain(X, y, 0)
    print(f"Information Gain: {gain:.4f}")
    print("\nDetailed Statistics:")
    print(f"Parent Entropy: {details['parent_entropy']:.4f}")
    print(f"Weighted Entropy: {details['weighted_entropy']:.4f}")
    print("\nSplit Information:")
    for value, info in details['split_info'].items():
        print(f"\nValue: {value}")
        print(f"Count: {info['count']}")
        print(f"Entropy: {info['entropy']:.4f}")
        print("Class Distribution:", dict(info['class_distribution']))
