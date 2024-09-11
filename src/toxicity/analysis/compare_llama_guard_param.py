import torch
from transformers import AutoModel
import numpy as np


def load_model(model_name):
    return AutoModel.from_pretrained(model_name)


def get_model_weights(model):
    return [p.data.cpu().numpy() for p in model.parameters()]


def calculate_weight_difference(weights1, weights2):
    total_diff = 0
    total_weights = 0

    for w1, w2 in zip(weights1, weights2):
        diff = np.sum(np.abs(w1 - w2))
        total_diff += diff
        total_weights += w1.size

    return total_diff / total_weights


def compare_models(model_names):
    models = [load_model(name) for name in model_names]
    weights = [get_model_weights(model) for model in models]

    results = {}
    for i in range(len(model_names)):
        for j in range(i + 1, len(model_names)):
            model1 = model_names[i]
            model2 = model_names[j]
            diff = calculate_weight_difference(weights[i], weights[j])
            results[f"{model1} vs {model2}"] = diff

    return results


model_names = [
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-Guard-2-8B"
]

differences = compare_models(model_names)

for comparison, diff in differences.items():
    print(f"Average weight difference for {comparison}: {diff}")

# Find which model Meta-Llama-Guard-2-8B is closer to
guard_vs_llama = differences["meta-llama/Meta-Llama-Guard-2-8B vs meta-llama/Meta-Llama-3-8B"]
guard_vs_instruct = differences["meta-llama/Meta-Llama-Guard-2-8B vs meta-llama/Meta-Llama-3-8B-Instruct"]

if guard_vs_llama < guard_vs_instruct:
    print("Meta-Llama-Guard-2-8B is closer to Meta-Llama-3-8B")
else:
    print("Meta-Llama-Guard-2-8B is closer to Meta-Llama-3-8B-Instruct")