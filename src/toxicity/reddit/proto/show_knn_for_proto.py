import fire
import torch
import torch.nn.functional as F
from tqdm import tqdm
import os

from toxicity.cpath import output_root_path
from toxicity.reddit.proto.protory_net2 import ProtoryNet3
from toxicity.reddit.path_helper import get_reddit_train_data_path
from toxicity.reddit.train_common import ClfDatasetLoader


def get_embeddings(model, dataset, batch_size=32):
    """Get embeddings for all examples in the dataset."""
    model.eval()
    all_embeddings = []

    # Create DataLoader
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False
    )

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            # Tokenize texts
            inputs = model.tokenizer(
                batch['text'],
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=512
            ).to(model.device)

            # Get embeddings from the model's encoder
            outputs = model.encode_inputs(
                inputs["input_ids"],
                inputs["attention_mask"])
            all_embeddings.append(outputs.cpu())  # Keep collecting on CPU to save GPU memory

    return torch.cat(all_embeddings, dim=0)


def find_nearest_examples(embeddings, prototypes, texts, device, k=5):
    """Find k nearest examples to each prototype vector."""
    # Move tensors to the specified device
    embeddings = embeddings.to(device)
    prototypes = prototypes.to(device)

    # Normalize embeddings and prototypes
    embeddings_normalized = F.normalize(embeddings, p=2, dim=1)
    prototypes_normalized = F.normalize(prototypes, p=2, dim=1)

    # Compute cosine similarities
    similarities = torch.mm(prototypes_normalized, embeddings_normalized.t())

    # Get top k nearest examples for each prototype
    top_k_similarities, top_k_indices = similarities.topk(k, dim=1)

    # Move results back to CPU for processing
    top_k_similarities = top_k_similarities.cpu()
    top_k_indices = top_k_indices.cpu()

    results = []
    for prototype_idx in range(len(prototypes)):
        prototype_results = []
        for j in range(k):
            example_idx = top_k_indices[prototype_idx][j].item()
            similarity = top_k_similarities[prototype_idx][j].item()
            prototype_results.append({
                'text': texts[example_idx],
                'similarity': similarity,
                'index': example_idx
            })
        results.append(prototype_results)

    return results


def main(sb = "NeutralPolitics"):
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    # Load model
    model = ProtoryNet3.from_pretrained(f"outputs/models/proto1_{sb}")
    model.to(device)
    prototypes = model.prototype_layer.prototypes

    # Load dataset
    dataset_loader = ClfDatasetLoader()
    dataset = dataset_loader.get(
        get_reddit_train_data_path(sb, "train")
    )

    # Get embeddings for all examples
    embeddings = get_embeddings(model, dataset)

    # Find nearest examples
    nearest_examples = find_nearest_examples(
        embeddings,
        prototypes,
        dataset['text'],
        device,
        k=5
    )

    save_path = os.path.join(
        output_root_path, "reddit", "proto_knn", f"{sb}.txt")

    # Open file for writing results
    with open(save_path, 'w', encoding='utf-8') as f:
        # Write results
        for prototype_idx, examples in enumerate(nearest_examples):
            f.write(f"\nPrototype {prototype_idx}: ==========\n")
            for i, example in enumerate(examples, 1):
                f.write(f"\n>> Nearest example {i} (similarity: {example['similarity']:.3f}):\n")
                # Truncate text if longer than 200 characters
                text = example['text'][:200] + "..." if len(example['text']) > 200 else example['text']
                f.write(f"  {text}\n")


if __name__ == "__main__":
    fire.Fire(main)
