from abc import ABC, abstractmethod
from typing import List
import numpy as np
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator



class TextClassifier(ABC):
    @abstractmethod
    def predict(self, texts: List[str]) -> List[int]:
        """
        Predict the class for a list of texts.

        Args:
            texts (List[str]): A list of text strings to classify.

        Returns:
            List[int]: A list of predicted class indices.
        """
        pass


class TextGenerator(ABC):
    @abstractmethod
    def predict(self, texts: List[str]) -> List[str]:
        """
        Predict the class for a list of texts.

        Args:
            texts (List[str]): A list of text strings to classify.

        Returns:
            List[int]: A list of predicted class indices.
        """
        pass


class RandomClassifier(TextClassifier):
    def __init__(self):
        pass

    def predict(self, texts: List[str]) -> List[int]:
        return np.random.randint(0, 2, len(texts)).tolist()


class LlamaGuard(TextGenerator):
    def __init__(self, model_id: str = "meta-llama/Meta-Llama-Guard-2-8B", use_toxicity=False):
        from llama_guard.load_llama_guard import load_llama_guard_model
        self.check_conversation: Callable[[List[str]], List[str]] = load_llama_guard_model(
            model_id,
            use_toxicity=use_toxicity)

    def predict(self, texts: List[str]) -> List[str]:
        predictions = []
        for text in texts:
            result = self.check_conversation([text])
            predictions.append(result)
        return predictions
