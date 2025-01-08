from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np


class TextSimilarity:
    def __init__(self, model_name='sentence-transformers/all-MiniLM-L6-v2'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def encode_sentences(self, sentences):
        inputs = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        outputs = self.model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return embeddings

    def compute_similarity(self, sentence1, sentence2):
        embeddings = self.encode_sentences([sentence1, sentence2])
        cosine_similarity = torch.nn.functional.cosine_similarity(embeddings[0], embeddings[1], dim=0)
        return cosine_similarity.item()

    def compute_similarity_matrix(self, list1, list2):
        embeddings1 = self.encode_sentences(list1)
        embeddings2 = self.encode_sentences(list2)

        # Normalize the embeddings
        embeddings1 = torch.nn.functional.normalize(embeddings1, p=2, dim=1)
        embeddings2 = torch.nn.functional.normalize(embeddings2, p=2, dim=1)

        similarity_matrix = torch.mm(embeddings1, embeddings2.T)
        return similarity_matrix.detach().numpy()  # Detach and convert to numpy array


def test():
    # Example usage
    text_sim = TextSimilarity()

    # Example sentences
    sentences_list1 = ["This is a sentence.", "How about this one?"]
    sentences_list2 = ["This is another sentence.", "Checking another similarity."]

    # Compute similarity matrix
    similarity_matrix = text_sim.compute_similarity_matrix(sentences_list1, sentences_list2)
    print(f"Similarity matrix:\n{similarity_matrix}")
