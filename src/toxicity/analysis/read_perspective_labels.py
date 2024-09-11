import json
import os
from typing import List, Tuple, Dict
from collections import Counter

from newbie.path_helper import get_open_ai_mod_perspective_api_res_path


def process_perspective_results(attributes: List[str], threshold: float, file_path: str) -> Dict[
    str, List[Tuple[str, int]]]:
    results = {attr: [] for attr in attributes}

    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                prompt = data['prompt']
                api_result = data['api_result']

                for attr in attributes:
                    if 'attributeScores' in api_result and attr in api_result['attributeScores']:
                        score = api_result['attributeScores'][attr]['summaryScore']['value']
                        label = 1 if score >= threshold else 0
                        results[attr].append((prompt, label))
                    else:
                        results[attr].append((prompt, 0))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
            except KeyError as e:
                print(f"Skipping line due to missing key: {e}")

    return results


def load_perspective_results(attribute: str, threshold: float, file_path: str) -> List[Tuple[str, int]]:
    results = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                data = json.loads(line)
                prompt = data['prompt']
                api_result = data['api_result']
                score = api_result['attributeScores'][attribute]['summaryScore']['value']
                label = 1 if score >= threshold else 0
                results.append((prompt, label))
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line}")
            except KeyError as e:
                print(f"Skipping line due to missing key: {e}")
    return results


def load_open_ai_mod_perspective_labels():
    load_perspective_results("TOXICITY", 0.5, get_open_ai_mod_perspective_api_res_path())


def count_distribution():
    attributes = ["PROFANITY", "TOXICITY", "INSULT", "THREAT", "IDENTITY_ATTACK", "SEVERE_TOXICITY"]
    threshold = 0.5
    file_path = get_open_ai_mod_perspective_api_res_path()

    processed_results = process_perspective_results(attributes, threshold, file_path)

    print(f"Processed {len(next(iter(processed_results.values())))} entries")

    for attr in attributes:
        true_count = sum(label for _, label in processed_results[attr])
        print(f"{attr}: {true_count} true labels")

    # Count items that are true for multiple attributes
    multi_attr_counts = Counter()
    for i in range(len(next(iter(processed_results.values())))):
        true_attrs = tuple(attr for attr in attributes if processed_results[attr][i][1] == 1)
        if true_attrs:
            multi_attr_counts[true_attrs] += 1

    print("\nItems true for multiple attributes:")
    for attrs, count in multi_attr_counts.most_common():
        print(f"{' & '.join(attrs)}: {count}")


# Example usage
if __name__ == "__main__":
    count_distribution()