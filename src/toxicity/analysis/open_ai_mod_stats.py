from datasets import load_dataset
from collections import Counter
from typing import Dict, Any
import json
from statistics import mean, median

def load_open`ai_dataset() -> Dict[str, Dict[str, Any]]:
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    return {str(i): {k:v for k,v in item.items() if k != "prompt"} for i, item in enumerate(openai_dataset)}

def analyze_label_statistics(openai_preds: Dict[str, Dict[str, Any]]):
    total_samples = len(openai_preds)
    label_counts = Counter()
    label_values = {}

    # First pass: count occurrences and collect all values
    for item in openai_preds.values():
        for label, value in item.items():
            if isinstance(value, (bool, int, float)):
                label_counts[label] += 1 if value else 0
                if label not in label_values:
                    label_values[label] = []
                label_values[label].append(value)

    # Print statistics
    print(f"Total samples: {total_samples}")
    print("\nLabel Statistics:")
    print("-" * 70)
    print("Label                          Count     Percentage  Avg Value")
    print("-" * 70)
    for label, count in label_counts.most_common():
        percentage = count / total_samples * 100
        avg_value = mean(label_values[label]) if label_values[label] else 'N/A'
        if isinstance(avg_value, float):
            avg_str = f"{avg_value:.4f}"
        else:
            avg_str = str(avg_value)
        print(f"{label:30} {count:10} {percentage:6.2f}%    {avg_str:10}")

# ... [rest of the code remains the same]

if __name__ == "__main__":
    openai_preds = load_openai_dataset()
    analyze_label_statistics(openai_preds)

"""
Label Statistics:
----------------------------------------------------------------------
Label                          Count     Percentage  Avg Value
----------------------------------------------------------------------
S                                     237  14.11%    0.2409    
H                                     162   9.64%    0.2101    
V                                      94   5.60%    0.0648    
S3                                     85   5.06%    0.0855    
HR                                     76   4.52%    0.0526    
SH                                     51   3.04%    0.0352    
H2                                     41   2.44%    0.0539    
V2                                     24   1.43%    0.0166    
"""