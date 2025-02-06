from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Tuple, Iterator, Optional


class NLIProcessor:
    """
    A class to process Natural Language Inference tasks using a pre-trained model.
    """

    def __init__(self,
                 model_name: str = 'cross-encoder/nli-deberta-v3-base',
                 batch_size: int = 32):
        """
        Initialize the NLI processor with a specific model.

        Args:
            model_name (str): The name of the pre-trained model to use
            batch_size (int): Number of pairs to process at once
        """
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.label_mapping = ['contradiction', 'entailment', 'neutral']
        self.batch_size = batch_size

    def _batch_pairs(self, sentence_pairs: List[Tuple[str, str]]) -> Iterator[List[Tuple[str, str]]]:
        """
        Split sentence pairs into batches.

        Args:
            sentence_pairs: List of all sentence pairs

        Returns:
            Iterator of batched sentence pairs
        """
        for i in range(0, len(sentence_pairs), self.batch_size):
            yield sentence_pairs[i:i + self.batch_size]

    def process_pairs(self,
                      sentence_pairs: List[Tuple[str, str]],
                      batch_size: Optional[int] = None) -> np.ndarray:
        if batch_size is not None:
            self.batch_size = batch_size

        all_probs = []
        self.model.eval()

        for batch in self._batch_pairs(sentence_pairs):
            premises = [pair[0] for pair in batch]
            hypotheses = [pair[1] for pair in batch]

            features = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                logits = self.model(**features).logits
                probs = F.softmax(logits, dim=-1)
                all_probs.append(probs.numpy())

        return np.concatenate(all_probs, axis=0)

    def get_predictions(self,
                        sentence_pairs: List[Tuple[str, str]],
                        batch_size: Optional[int] = None) -> List[str]:
        probs = self.process_pairs(sentence_pairs, batch_size=batch_size)
        predictions = [self.label_mapping[idx] for idx in np.argmax(probs, axis=1)]
        return predictions

    def get_predictions_ex(self,
                        sentence_pairs: List[Tuple[str, str]],
                        batch_size: Optional[int] = None) -> List[str]:
        probs = self.process_pairs(sentence_pairs, batch_size=batch_size)
        predictions = [self.label_mapping[idx] for idx in np.argmax(probs, axis=1)]
        max_indices = np.argmax(probs, axis=1)
        confidence = probs[np.arange(len(probs)), max_indices]
        return predictions, confidence

# Example usage:
if __name__ == "__main__":
    # Initialize the processor with custom batch size
    nli_processor = NLIProcessor(batch_size=16)

    # Example sentence pairs
    pairs = [
        ("A man is eating pizza", "A man eats something"),
        ("A black race car starts up in front of a crowd of people.",
         "A man is driving down a lonely road.")
    ]

    # Get probabilities
    probs = nli_processor.process_pairs(pairs)
    print("Probabilities shape:", probs.shape)
    print("Probabilities:", probs)
    print("Sum of probabilities for each pair:", probs.sum(axis=1))

    # Get predictions
    predictions = nli_processor.get_predictions_ex(pairs)
    print("Predictions:", predictions)