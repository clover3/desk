import sys

# TODO randomly select instances, and ask GPT how they differ.

from datasets import load_dataset
import random

from llama_user.llama_helper.lf_client import LLMClient


def randomly_sample_pos_neg():
    openai_dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    openai_dataset.shuffle()

    pos_texts = []
    neg_texts = []

    for item in openai_dataset:
        labels = {k: v for k, v in item.items() if k != "prompt"}
        any_true = any(labels.values())
        text = item["prompt"]
        text = text[:300]
        if any_true:
            pos_texts.append(text)
        else:
            neg_texts.append(text)

    random.shuffle(pos_texts)
    random.shuffle(neg_texts)
    return concat_pos_neg(pos_texts, neg_texts)


def concat_pos_neg(pos_texts, neg_texts):
    lines = []
    for idx, text in enumerate(pos_texts):
        lines.append(f"<positive {idx + 1}>: {text} </positive {idx + 1}>")
    for idx, text in enumerate(neg_texts):
        lines.append(f"<negative {idx + 1}>{idx + 1}: {text} </negative {idx + 1}>")
    return "\n".join(lines)


def main():
    sample_text = randomly_sample_pos_neg()
    instruction = "Identify what are the criteria for being classified to be positive compared to negative classified ones."

    prompt = sample_text + "===\n" + instruction
    response = LLMClient().ask(prompt)
    print(response)



if __name__ == "__main__":
    main()
