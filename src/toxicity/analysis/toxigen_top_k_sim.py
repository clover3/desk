from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
import nltk
from nltk.tokenize import sent_tokenize

from io_helper import load_predictions
from dataset_helper.load_toxigen import ToxigenBinary
from path_helper import get_dataset_pred_save_path


# Download the punkt tokenizer for sentence splitting


def read_texts(n):
    """Read N texts from user input."""
    texts = []
    for i in range(n):
        print(f"Enter text {i + 1}:")
        texts.append(input())
    return texts


def encode_texts(texts, method='tfidf'):
    """Encode texts using the specified method."""
    if method == 'tfidf':
        nltk.download('punkt', quiet=True)
        vectorizer = TfidfVectorizer()
        return vectorizer.fit_transform(texts)
    elif method == 'bert':
        model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        return model.encode(texts)
    else:
        raise ValueError("Unsupported encoding method")


def split_into_sentences(texts):
    """Split texts into sentences."""
    all_sentences = []
    for text in texts:
        sentences = sent_tokenize(text)
        all_sentences.extend(sentences)
    return all_sentences


def find_top_k_similar(encodings, n, k, threshold=0.5):
    """Find top-k similar sentences for first n items with similarity scores."""
    similarities = cosine_similarity(encodings[:n], encodings)

    top_k_indices = []
    top_k_scores = []

    for i in range(n):
        # Sort similarities for the current sentence
        sorted_indices = similarities[i].argsort()[::-1]
        sorted_scores = similarities[i][sorted_indices]

        # Filter based on threshold and exclude self-similarity
        filtered_indices = [idx for idx, score in zip(sorted_indices, sorted_scores)
                            if score >= threshold and idx != i][:k]
        filtered_scores = [score for idx, score in zip(sorted_indices, sorted_scores)
                           if score >= threshold and idx != i][:k]

        top_k_indices.append(filtered_indices)
        top_k_scores.append(filtered_scores)

    return top_k_indices, top_k_scores


def get_llama_guard_labels():
    run_name = "llama_guard2_prompt"
    dataset_name: str = 'toxigen'
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    ids, text_predictions = load_predictions(save_path)

    target_string = "S1"
    llama_guard_preds: list[int] = [1 if target_string in pred else 0 for pred in text_predictions]
    return llama_guard_preds



def main():
    # 1. Load dataset
    dataset = ToxigenBinary("test")
    sentences = [e['text'] for e in dataset]
    labels = [e['label'] for e in dataset]
    lg_labels = get_llama_guard_labels()

    # 2. Encode texts
    encodings = encode_texts(sentences, method="bert")

    # 3. Find top-k similar sentences for first n items
    n = 40
    k = 5
    threshold = 0.6  # Set your desired similarity threshold here
    top_k_indices, top_k_scores = find_top_k_similar(encodings, n, k, threshold)

    # Print results
    for i in range(n):
        if top_k_indices[i]:
            print(f"\nTop similar sentences for sentence {i + 1}:")
            print("k\tToxigen\tLlama\tsimilarity\ttext")
            print(f"Orig\t{labels[i]}\t{lg_labels[i]}\t-\t{sentences[i]}")
            for j, (idx, score) in enumerate(zip(top_k_indices[i], top_k_scores[i])):
                print(f"{j + 1}\t{labels[idx]}\t{lg_labels[idx]}\t{score:.4f}\t{sentences[idx]}")


if __name__ == "__main__":
    main()