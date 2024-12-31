
# TODO randomly select instances, and ask GPT how they differ.

from datasets import load_dataset
import random

def load_openai_mod():
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    openai_dataset.shuffle()

    pos_item = []
    neg_item = []

    for item in openai_dataset:
        labels = {k: v for k, v in item.items() if k != "prompt"}
        any_true = any(labels.values())
        text = item["prompt"]
        if any_true:
            pos_item.append(item)
        else:
            neg_item.append(item)
    random.shuffle(pos_item)
    random.shuffle(neg_item)
    print("Positive")
    for idx in range(10):
        text = pos_item[idx]["prompt"]
        other_labels = {k:v for k, v in pos_item[idx].items() if k != "prompt"}
        print(f"{idx+1}: {text}")

    print("Negative")
    for idx in range(10):
        text = neg_item[idx]["prompt"]
        other_labels = {k:v for k, v in pos_item[idx].items() if k != "prompt"}
        print(f"{idx+1}: {text}")


def main():
    load_openai_mod()


if __name__ == "__main__":
    main()
