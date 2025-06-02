import numpy as np
from collections import Counter, defaultdict
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report
from typing import List, Tuple
import re
import pickle
import json
import os

from desk_util.clf_util import eval_prec_recall_f1_acc


class NgramLogisticRegression:
    def __init__(self, n=[1, 2], max_features=10000, random_state=42):
        """
        N-gram Logistic Regression Model

        Args:
            n: N-gram sizes as list (e.g., [1, 2, 3] for unigrams, bigrams, and trigrams)
            max_features: Maximum number of features to keep
            random_state: Random state for reproducibility
        """
        self.n = n if isinstance(n, list) else [n]
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = None
        self.model = None
        self.is_fitted = False

    def _create_ngrams(self, tokens: List[str]) -> List[str]:
        """Create n-grams from a list of tokens for all specified n values"""
        all_ngrams = []

        for n_size in self.n:
            if n_size == 1:
                all_ngrams.extend(tokens)
            else:
                for i in range(len(tokens) - n_size + 1):
                    ngram = ' '.join(tokens[i:i + n_size])
                    all_ngrams.append(ngram)

        return all_ngrams

    def _preprocess_data(self, data: List[Tuple[List[str], int]]):
        """Convert tokenized texts to n-gram strings"""
        texts = []
        labels = []

        for tokens, label in data:
            # Create n-grams from tokens
            ngrams = self._create_ngrams(tokens)
            # Join n-grams into a single string for vectorization
            text = ' '.join(ngrams)
            texts.append(text)
            labels.append(label)

        return texts, np.array(labels)

    def fit(self, train_data: List[Tuple[List[str], int]]):
        """
        Train the n-gram logistic regression model

        Args:
            train_data: List of (tokenized_text, label) tuples
        """
        # Preprocess the data
        texts, labels = self._preprocess_data(train_data)

        # Create and fit the vectorizer
        self.vectorizer = CountVectorizer(
            max_features=self.max_features,
            token_pattern=r'\S+',  # Split on whitespace
            lowercase=False  # Assume tokens are already preprocessed
        )

        # Transform texts to feature vectors
        X = self.vectorizer.fit_transform(texts)

        # Train logistic regression model
        self.model = LogisticRegression(
            random_state=self.random_state,
            max_iter=1000
        )
        self.model.fit(X, labels)

        self.is_fitted = True
        return self

    def predict(self, test_data: List[Tuple[List[str], int]]) -> np.ndarray:
        """
        Make predictions on test data

        Args:
            test_data: List of (tokenized_text, label) tuples

        Returns:
            Array of predicted labels
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        texts, _ = self._preprocess_data(test_data)
        X = self.vectorizer.transform(texts)
        return self.model.predict(X)

    def predict_proba(self, test_data: List[Tuple[List[str], int]]) -> np.ndarray:
        """
        Get prediction probabilities

        Args:
            test_data: List of (tokenized_text, label) tuples

        Returns:
            Array of prediction probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")

        texts, _ = self._preprocess_data(test_data)
        X = self.vectorizer.transform(texts)
        return self.model.predict_proba(X)

    def evaluate(self, test_data: List[Tuple[List[str], int]]) -> dict:
        """
        Evaluate model performance on test data

        Args:
            test_data: List of (tokenized_text, label) tuples

        Returns:
            Dictionary with evaluation metrics
        """
        texts, true_labels = self._preprocess_data(test_data)
        predictions = self.predict(test_data)
        metrics = eval_prec_recall_f1_acc(true_labels, predictions)
        return metrics


    def get_feature_importance(self, top_n=20) -> List[Tuple[str, float]]:
        """
        Get the most important features (n-grams) for classification

        Args:
            top_n: Number of top features to return

        Returns:
            List of (feature, importance) tuples
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")

        feature_names = self.vectorizer.get_feature_names_out()

        # For binary classification, use coefficients directly
        # For multiclass, use the maximum absolute coefficient across classes
        if len(self.model.coef_) == 1:
            coefficients = self.model.coef_[0]
        else:
            coefficients = np.max(np.abs(self.model.coef_), axis=0)

        # Get indices of top features
        top_indices = np.argsort(np.abs(coefficients))[-top_n:][::-1]

        top_features = [(feature_names[i], coefficients[i]) for i in top_indices]
        return top_features

    def save(self, filepath: str):
        """
        Save the trained model to disk

        Args:
            filepath: Path to save the model (without extension)
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)

        # Save model parameters and components
        model_data = {
            'n': self.n,
            'max_features': self.max_features,
            'random_state': self.random_state,
            'is_fitted': self.is_fitted
        }

        # Save metadata as JSON
        with open(f"{filepath}_metadata.json", 'w') as f:
            json.dump(model_data, f, indent=2)

        # Save vectorizer
        with open(f"{filepath}_vectorizer.pkl", 'wb') as f:
            pickle.dump(self.vectorizer, f)

        # Save logistic regression model
        with open(f"{filepath}_model.pkl", 'wb') as f:
            pickle.dump(self.model, f)

        print(f"Model saved successfully to {filepath}_*")

    def load(self, filepath: str):
        """
        Load a trained model from disk

        Args:
            filepath: Path to load the model from (without extension)
        """
        try:
            # Load metadata
            with open(f"{filepath}_metadata.json", 'r') as f:
                model_data = json.load(f)

            # Restore model parameters
            self.n = model_data['n']
            self.max_features = model_data['max_features']
            self.random_state = model_data['random_state']
            self.is_fitted = model_data['is_fitted']

            # Load vectorizer
            with open(f"{filepath}_vectorizer.pkl", 'rb') as f:
                self.vectorizer = pickle.load(f)

            # Load logistic regression model
            with open(f"{filepath}_model.pkl", 'rb') as f:
                self.model = pickle.load(f)

            print(f"Model loaded successfully from {filepath}_*")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Model files not found at {filepath}. Make sure all required files exist: "
                                    f"{filepath}_metadata.json, {filepath}_vectorizer.pkl, {filepath}_model.pkl")
        except Exception as e:
            raise Exception(f"Error loading model: {str(e)}")

    @classmethod
    def load_model(cls, filepath: str) -> 'NgramLogisticRegression':
        """
        Class method to create a new instance and load a model

        Args:
            filepath: Path to load the model from (without extension)

        Returns:
            NgramLogisticRegression instance with loaded model
        """
        instance = cls()
        instance.load(filepath)
        return instance


# Example usage
if __name__ == "__main__":
    # Sample data
    train_data = [
        (['this', 'is', 'a', 'positive', 'example'], 1),
        (['great', 'movie', 'loved', 'it'], 1),
        (['excellent', 'service', 'highly', 'recommend'], 1),
        (['this', 'is', 'a', 'negative', 'example'], 0),
        (['terrible', 'movie', 'waste', 'of', 'time'], 0),
        (['poor', 'service', 'very', 'disappointed'], 0),
        (['amazing', 'product', 'best', 'purchase'], 1),
        (['awful', 'experience', 'never', 'again'], 0),
    ]

    test_data = [
        (['good', 'movie', 'recommend', 'it'], 1),
        (['bad', 'service', 'disappointed'], 0),
        (['excellent', 'product', 'love', 'it'], 1),
        (['terrible', 'experience', 'waste'], 0),
    ]

    # Test different n-gram combinations
    n_gram_configs = [[1], [2], [1, 2], [1, 2, 3]]

    for n_list in n_gram_configs:
        print(f"\n=== {n_list}-gram Model ===")

        # Create and train model
        model = NgramLogisticRegression(n=n_list)
        model.fit(train_data)

        # Make predictions
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)

        print(f"Predictions: {predictions}")
        print(f"Probabilities: {probabilities}")

        # Evaluate model
        results = model.evaluate(test_data)
        print(f"Accuracy: {results['accuracy']:.2f}")

        # Show top features
        top_features = model.get_feature_importance(top_n=5)
        print("Top features:")
        for feature, importance in top_features:
            print(f"  {feature}: {importance:.3f}")

    # Demonstrate save/load functionality
    print("\n=== Save/Load Demo ===")

    # Train a model
    model = NgramLogisticRegression(n=[1, 2])
    model.fit(train_data)

    # Save the model
    model.save("./saved_model")

    # Load the model in a new instance
    loaded_model = NgramLogisticRegression.load_model("./saved_model")

    # Test that loaded model works
    loaded_predictions = loaded_model.predict(test_data)
    print(f"Loaded model predictions: {loaded_predictions}")

    # Verify predictions are the same
    original_predictions = model.predict(test_data)
    print(f"Original predictions match loaded: {np.array_equal(original_predictions, loaded_predictions)}")

    # Alternative loading method
    model2 = NgramLogisticRegression()
    model2.load("./saved_model")
    print(f"Alternative load method works: {np.array_equal(original_predictions, model2.predict(test_data))}")