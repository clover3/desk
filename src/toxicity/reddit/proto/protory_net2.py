import numpy as np
import tqdm
from dataclasses import dataclass
from sklearn_extra.cluster import KMedoids

import torch
from torch.nn import BCELoss, BCEWithLogitsLoss
from transformers import BertPreTrainedModel, PreTrainedModel, AutoConfig, PretrainedConfig
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from typing import List, Dict, Tuple, Optional, Union

from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.utils import ModelOutput

from toxicity.reddit.proto.protory_net_torch import PrototypeLayer, DistanceLayer


@dataclass
class ProtoOutput(ModelOutput):
    logits: torch.FloatTensor = None
    distances: torch.FloatTensor = None
    prototypes: torch.FloatTensor = None


@dataclass
class ProtoSequenceClassifierOutput(SequenceClassifierOutput):
    distances: torch.FloatTensor = None
    prototypes: torch.FloatTensor = None


@dataclass
class ProtoryConfig:
    k_protos: int = 10
    alpha: float = 0.0001
    beta: float = 0.01
    lstm_dim: int = 128
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class ProtoryNet2(BertPreTrainedModel):
    def __init__(self, p_config: Optional[ProtoryConfig] = None):

        if p_config is None:
            p_config = ProtoryConfig()

        base_config = AutoConfig.from_pretrained(
            p_config.base_model_name, trust_remote_code=True
        )
        super().__init__(base_config)

        k_protos = p_config.k_protos
        self.k_protos = k_protos
        self.alpha = p_config.alpha
        self.beta = p_config.beta
        self.mapped_prototypes = {}

        # Initialize sentence encoder
        self.tokenizer = AutoTokenizer.from_pretrained(p_config.base_model_name)
        self.model = AutoModel.from_pretrained(p_config.base_model_name)
        self.vect_size = self.model.config.hidden_size

        # Initialize layers
        self.prototype_layer = PrototypeLayer(k_protos, self.vect_size)
        self.distance_layer = DistanceLayer()
        self.lstm = nn.LSTM(k_protos, p_config.lstm_dim, batch_first=True)
        self.classifier = nn.Linear(p_config.lstm_dim, 1)
        self.use_grad = False

    def encode_inputs(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.use_grad):
            model_output = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Tokenize and encode sentences
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        proto_out = self.apply_proto_classifier(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = self.compute_loss(proto_out, labels)
        if not return_dict:
            probs = torch.sigmoid(proto_out.logits)
            output = (probs,)
            if loss is not None:
                ret = ((loss,) + output)
            else:
                ret = output
            return ret

        return SequenceClassifierOutput(
            loss=loss,
            logits=proto_out.logits,
            # prototypes=proto_out.prototypes,
            # distances=proto_out.distances
        )

    def apply_proto_classifier(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        sentence_embeddings = self.encode_inputs(input_ids, attention_mask)
        sentence_embeddings = sentence_embeddings.unsqueeze(1)  # [batch_size, 1, vect_size]
        distances, prototypes = self.prototype_layer(sentence_embeddings)
        dist_hot_vect = self.distance_layer(distances)
        lstm_out, (h_n, c_n) = self.lstm(dist_hot_vect)
        logits = self.classifier(lstm_out[:, -1, :])
        return ProtoOutput(logits, distances, prototypes)

    def compute_loss(self,
                     proto_out: ProtoOutput,
                     targets: torch.Tensor) -> torch.Tensor:
        loss_fct = BCEWithLogitsLoss()
        targets = targets.unsqueeze(1).float()
        bce_loss = loss_fct(proto_out.logits, targets)

        clustering_loss = torch.sum(torch.min(proto_out.distances, dim=1)[0])

        proto_distances = torch.cdist(proto_out.prototypes, proto_out.prototypes)
        mask = torch.eye(self.k_protos, device=proto_out.prototypes.device)
        proto_distances = proto_distances + mask * torch.max(proto_distances)
        separation_loss = torch.sigmoid(torch.min(proto_distances.view(-1))) + 1e-8

        # Combine losses
        total_loss = bce_loss + self.alpha * clustering_loss + self.beta * separation_loss
        return total_loss

    def init_prototypes(self, sentence_embeddings_all):
        kmedoids = KMedoids(n_clusters=self.k_protos, random_state=0).fit(sentence_embeddings_all)
        k_cents = kmedoids.cluster_centers_
        self.prototype_layer.set_prototypes(k_cents)

    def projection(self, sample_sentences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        sample_embeddings = []

        # Get embeddings for sample sentences
        for i in range(0, len(sample_sentences), 32):
            batch = sample_sentences[i:min(i + 32, len(sample_sentences))]
            with torch.no_grad():
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = self.model(**encoded)
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
                output = self.model(**encoded)
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


@dataclass
class ProtoryConfig:
    k_protos: int = 10
    alpha: float = 0.0001
    beta: float = 0.01
    lstm_dim: int = 128
    base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"


class ProtoryConfig2(PretrainedConfig):
    def __init__(
        self,
        k_protos=10,
        alpha: float = 0.0001,
        beta: float = 0.01,
        lstm_dim: int = 128,
        **kwargs,
    ):
        self.k_protos = k_protos
        self.alpha = alpha
        self.beta = beta
        self.lstm_dim = lstm_dim
        self.base_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
        super().__init__(**kwargs)


class ProtoryNet3(BertPreTrainedModel):
    config_class = ProtoryConfig2
    def __init__(self, p_config: Optional[ProtoryConfig2]):
        print(p_config)
        super().__init__(p_config)

        k_protos = p_config.k_protos
        self.k_protos = k_protos
        self.alpha = p_config.alpha
        self.beta = p_config.beta
        self.mapped_prototypes = {}

        # Initialize sentence encoder
        self.tokenizer = AutoTokenizer.from_pretrained(p_config.base_model_name)
        self.model = AutoModel.from_pretrained(p_config.base_model_name)
        self.vect_size = self.model.config.hidden_size

        # Initialize layers
        self.prototype_layer = PrototypeLayer(k_protos, self.vect_size)
        self.distance_layer = DistanceLayer()
        self.lstm = nn.LSTM(k_protos, p_config.lstm_dim, batch_first=True)
        self.classifier = nn.Linear(p_config.lstm_dim, 1)
        self.use_grad = False

    def encode_inputs(self, input_ids, attention_mask):
        with torch.set_grad_enabled(self.use_grad):
            model_output = self.model(input_ids, attention_mask)
        sentence_embeddings = self.mean_pooling(model_output, attention_mask)
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)
        return sentence_embeddings

    def mean_pooling(self, model_output: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    def forward(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        # Tokenize and encode sentences
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        proto_out = self.apply_proto_classifier(input_ids, attention_mask)
        loss = None
        if labels is not None:
            loss = self.compute_loss(proto_out, labels)
        if not return_dict:
            probs = torch.sigmoid(proto_out.logits)
            output = (probs,)
            if loss is not None:
                ret = ((loss,) + output)
            else:
                ret = output
            return ret

        return SequenceClassifierOutput(
            loss=loss,
            logits=proto_out.logits,
            # prototypes=proto_out.prototypes,
            # distances=proto_out.distances
        )

    def apply_proto_classifier(
            self,
            input_ids: Optional[torch.Tensor] = None,
            attention_mask: Optional[torch.Tensor] = None,
    ):
        sentence_embeddings = self.encode_inputs(input_ids, attention_mask)
        sentence_embeddings = sentence_embeddings.unsqueeze(1)  # [batch_size, 1, vect_size]
        distances, prototypes = self.prototype_layer(sentence_embeddings)
        dist_hot_vect = self.distance_layer(distances)
        lstm_out, (h_n, c_n) = self.lstm(dist_hot_vect)
        logits = self.classifier(lstm_out[:, -1, :])
        return ProtoOutput(logits, distances, prototypes)

    def compute_loss(self,
                     proto_out: ProtoOutput,
                     targets: torch.Tensor) -> torch.Tensor:
        loss_fct = BCEWithLogitsLoss()
        targets = targets.unsqueeze(1).float()
        bce_loss = loss_fct(proto_out.logits, targets)

        clustering_loss = torch.sum(torch.min(proto_out.distances, dim=1)[0])

        proto_distances = torch.cdist(proto_out.prototypes, proto_out.prototypes)
        mask = torch.eye(self.k_protos, device=proto_out.prototypes.device)
        proto_distances = proto_distances + mask * torch.max(proto_distances)
        separation_loss = torch.sigmoid(torch.min(proto_distances.view(-1))) + 1e-8

        # Combine losses
        total_loss = bce_loss + self.alpha * clustering_loss + self.beta * separation_loss
        return total_loss

    def init_prototypes(self, sentence_embeddings_all):
        kmedoids = KMedoids(n_clusters=self.k_protos, random_state=0).fit(sentence_embeddings_all)
        k_cents = kmedoids.cluster_centers_
        self.prototype_layer.set_prototypes(k_cents)

    def projection(self, sample_sentences: List[str]) -> torch.Tensor:
        device = next(self.parameters()).device
        sample_embeddings = []

        # Get embeddings for sample sentences
        for i in range(0, len(sample_sentences), 32):
            batch = sample_sentences[i:min(i + 32, len(sample_sentences))]
            with torch.no_grad():
                encoded = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt")
                encoded = {k: v.to(device) for k, v in encoded.items()}
                output = self.model(**encoded)
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
                output = self.model(**encoded)
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
