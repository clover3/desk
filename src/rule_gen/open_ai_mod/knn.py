from typing import Optional

import fire
import torch
import torch.nn.functional as F
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader
from transformers import AutoModel, AutoTokenizer, Trainer

from rule_gen.open_ai_mod.train_proto import get_data_arguments
from rule_gen.reddit.proto.train_proto_reddit import apply_tokenize, get_tokenize_formatter
from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg, get_default_training_argument


class SentenceEncoder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initialize the sentence encoder with a pre-trained model

        Args:
            model_name: Name of the pre-trained model to use
        """
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.vect_size = self.model.config.hidden_size

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def encode(self,
               input_ids: Optional[torch.Tensor] = None,
               attention_mask: Optional[torch.Tensor] = None,
               normalize: bool = True) -> torch.Tensor:
        """
        Encode input sequences to embeddings

        Args:
            input_ids: Tokenized input sequences
            attention_mask: Attention mask for the input sequences
            normalize: Whether to normalize the output embeddings

        Returns:
            Encoded sentence embeddings
        """
        with torch.no_grad():
            model_output = self.model(input_ids, attention_mask)

        sentence_embeddings = self.mean_pooling(model_output, attention_mask)

        if normalize:
            sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    def encode_batch(self, dataset):
        class EncoderWrapper(nn.Module):
            def __init__(self, original_model):
                super().__init__()
                self.model = original_model

            def forward(self_inner, input_ids, attention_mask):
                return self.encode(input_ids, attention_mask)

        training_args = get_default_training_argument("run_name")
        encoder = EncoderWrapper(self.model)
        trainer = Trainer(model=encoder,
                          args=training_args,
                          )
        outputs = trainer.predict(dataset)
        sentence_embeddings_all = outputs.predictions
        return sentence_embeddings_all


    def extract_labels(self, dataloader) -> torch.Tensor:
        """
        Extract labels from the dataloader

        Args:
            dataloader: PyTorch DataLoader containing batches with labels

        Returns:
            Tensor of all labels concatenated
        """
        all_labels = []
        for item in dataloader:
            labels = item['labels']
            all_labels.append(labels)

        return all_labels

    def encode_and_label(self, dataloader):
        embeddings = self.encode_batch(dataloader)
        labels = self.extract_labels(dataloader)
        return embeddings, labels


def main(debug=False):
    encoder = SentenceEncoder()
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(debug)
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    tokenize_format = get_tokenize_formatter(encoder.tokenizer, dataset_args.max_length)
    tokenized_train, tokenized_eval = apply_tokenize(
        train_dataset, eval_dataset, tokenize_format)

    # Encode and get labels using the DataLoader
    X_train, y_train = encoder.encode_and_label(tokenized_train)

    pipeline = Pipeline([
        ('scaler', StandardScaler()),  # Scale features
        ('knn', KNeighborsClassifier(n_neighbors=3, weights='distance', n_jobs=-1))
    ])

    scores = cross_val_score(pipeline, X_train, y_train, cv=5)
    print(f"Cross-validation scores: {scores}")
    print(f"Average accuracy: {scores.mean():.2f} ± {scores.std() * 2:.2f}")


if __name__ == "__main__":
    fire.Fire(main)
