import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional


class PrototypeLayer(nn.Module):
    def __init__(self, n_protos: int, vect_size: int, initial_prototypes: Optional[torch.Tensor] = None):
        super().__init__()
        self.n_protos = n_protos
        self.vect_size = vect_size

        if initial_prototypes is not None:
            self.prototypes = nn.Parameter(initial_prototypes)
        else:
            self.prototypes = nn.Parameter(torch.randn(n_protos, vect_size))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x shape: [batch_size, seq_len, vect_size]
        # Expand dimensions for broadcasting
        x_expanded = x.unsqueeze(2)  # [batch_size, seq_len, 1, vect_size]

        protos_expanded = self.prototypes.unsqueeze(0).unsqueeze(0)  # [1, 1, n_protos, vect_size]

        # Calculate distances
        distances = torch.sum((x_expanded - protos_expanded) ** 2, dim=-1)  # [batch_size, seq_len, n_protos]
        return distances, self.prototypes


class DistanceLayer(nn.Module):
    def __init__(self, alpha: float = 0.1, beta: float = 1e6):
        super().__init__()
        self.alpha = alpha
        self.beta = beta

    def forward(self, distances: torch.Tensor) -> torch.Tensor:
        # Softmax for minimum distance selection
        min_dist_ind = F.softmax(-distances * self.beta, dim=-1)
        # Exponential distance function
        e_dist = torch.exp(-self.alpha * distances) + 1e-8
        # Combine both
        dist_hot_vect = min_dist_ind * e_dist
        return dist_hot_vect


class ProtoryNet(nn.Module):
    def __init__(self, k_protos: int = 10, alpha: float = 0.0001, beta: float = 0.01,
                 model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        super().__init__()
        self.k_protos = k_protos
        self.alpha = alpha
        self.beta = beta
        self.mapped_prototypes = {}

        # Initialize sentence encoder
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name)
        self.vect_size = self.encoder.config.hidden_size

        # Initialize layers
        self.prototype_layer = PrototypeLayer(k_protos, self.vect_size)
        self.distance_layer = DistanceLayer()
        self.lstm = nn.LSTM(k_protos, 128, batch_first=True)
        self.classifier = nn.Linear(128, 1)

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(self, inputs: List[str]) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Tokenize and encode sentences
        encoded = self.tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
        encoded = {k: v.to(next(self.parameters()).device) for k, v in encoded.items()}

        with torch.no_grad():
            model_output = self.encoder(**encoded)

        # Get sentence embeddings
        sentence_embeddings = self.mean_pooling(model_output, encoded['attention_mask'])
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        # Add sequence dimension
        sentence_embeddings = sentence_embeddings.unsqueeze(1)  # [batch_size, 1, vect_size]
        # Get prototype distances and similarities
        distances, prototypes = self.prototype_layer(sentence_embeddings)
        dist_hot_vect = self.distance_layer(distances)

        # Process through LSTM
        lstm_out, (h_n, c_n) = self.lstm(dist_hot_vect)

        # Final classification
        logits = self.classifier(lstm_out[:, -1, :])
        predictions = torch.sigmoid(logits)

        return predictions.squeeze(), distances, prototypes

    def compute_loss(self, predictions: torch.Tensor, targets: torch.Tensor,
                     distances: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        # Binary cross entropy loss
        targets = targets.float()
        bce_loss = F.binary_cross_entropy(predictions, targets)

        # Prototype clustering loss
        clustering_loss = torch.sum(torch.min(distances, dim=1)[0])

        # Prototype separation loss
        proto_distances = torch.cdist(prototypes, prototypes)
        mask = torch.eye(self.k_protos, device=prototypes.device)
        proto_distances = proto_distances + mask * torch.max(proto_distances)
        separation_loss = torch.sigmoid(torch.min(proto_distances.view(-1))) + 1e-8

        # Combine losses
        total_loss = bce_loss + self.alpha * clustering_loss + self.beta * separation_loss
        return total_loss

    def projection(self, sample_sentences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        sample_embeddings = []

        # Get embeddings for sample sentences
        for i in range(0, len(sample_sentences), 32):
            batch = sample_sentences[i:min(i + 32, len(sample_sentences))]
            with torch.no_grad():
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = self.encoder(**encoded)
                embeddings = self.mean_pooling(output, encoded['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                sample_embeddings.append(embeddings)

        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        # Find closest sentences to prototypes
        prototypes = self.prototype_layer.prototypes.detach()
        distances = torch.cdist(prototypes, sample_embeddings)
        closest_indices = torch.argmin(distances, dim=1)

        new_prototypes = sample_embeddings[closest_indices]
        return new_prototypes

    def show_prototypes(self, sample_sentences: List[str], k_closest: int = 10) -> Dict[int, str]:
        device = next(self.parameters()).device
        sample_embeddings = []

        # Get embeddings for sample sentences
        for i in range(0, len(sample_sentences), 32):
            batch = sample_sentences[i:min(i + 32, len(sample_sentences))]
            with torch.no_grad():
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = self.encoder(**encoded)
                embeddings = self.mean_pooling(output, encoded['attention_mask'])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                sample_embeddings.append(embeddings)

        sample_embeddings = torch.cat(sample_embeddings, dim=0)

        # Calculate distances to prototypes
        prototypes = self.prototype_layer.prototypes.detach()
        distances = torch.cdist(prototypes, sample_embeddings)

        # Find k closest sentences for each prototype
        closest_indices = torch.topk(distances, k=k_closest, dim=1, largest=False)

        # Map prototypes to their closest sentences
        self.mapped_prototypes = {}
        for i in range(self.k_protos):
            self.mapped_prototypes[i] = sample_sentences[closest_indices.indices[i, 0]]
            print(f"Prototype {i}: {self.mapped_prototypes[i]}")

        return self.mapped_prototypes

    def save_model(self, path: str):
        torch.save({
            'model_state_dict': self.state_dict(),
            'k_protos': self.k_protos,
            'vect_size': self.vect_size,
            'alpha': self.alpha,
            'beta': self.beta,
            'mapped_prototypes': self.mapped_prototypes
        }, path)

    @classmethod
    def load_model(cls, path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        checkpoint = torch.load(path, map_location=device)
        model = cls(
            k_protos=checkpoint['k_protos'],
            alpha=checkpoint['alpha'],
            beta=checkpoint['beta']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.mapped_prototypes = checkpoint['mapped_prototypes']
        return model



# Training helper functions
def train_epoch(model: ProtoryNet,
                train_loader: DataLoader,
                optimizer: torch.optim.Optimizer,) -> float:
    model.train()
    total_loss = 0

    for batch_items in tqdm(train_loader):
        optimizer.zero_grad()
        predictions, distances, prototypes = model(batch_items["text"])
        batch_labels = batch_items["label"]
        loss = model.compute_loss(predictions, batch_labels, distances, prototypes)

        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def evaluate(model: ProtoryNet,
             eval_loader: DataLoader) -> Tuple[float, float]:
    model.eval()
    correct = 0
    total = 0
    total_loss = 0

    with torch.no_grad():
        for batch in eval_loader:
            batch_texts = batch["text"]
            batch_labels = batch["label"]

            predictions, distances, prototypes = model(batch_texts)
            loss = model.compute_loss(predictions, batch_labels, distances, prototypes)

            predictions = (predictions > 0.5).float()
            correct += (predictions == batch_labels).sum().item()
            total += len(batch_labels)
            total_loss += loss.item()

    accuracy = correct / total
    avg_loss = total_loss / len(eval_loader)
    return accuracy, avg_loss