from typing import List, Optional, Tuple

import torch
from torch import nn as nn
from transformers import BertForSequenceClassification, BertPreTrainedModel


class BertProbe(BertForSequenceClassification):
    def __init__(
            self,
            config,
            hidden_size: int = 768,
            num_classes: int = 2,
            layers_to_probe: List[int] = None,
    ):
        super().__init__(config)
        self.num_layers = self.bert.config.num_hidden_layers
        self.hidden_size = hidden_size

        # If no specific layers are specified, probe all layers
        if layers_to_probe is None:
            self.layers_to_probe = list(range(self.num_layers + 1))  # +1 for embeddings
        else:
            self.layers_to_probe = layers_to_probe

        # Freeze BERT parameters
        self.bert.eval()  # Set BERT to evaluation mode
        for param in self.bert.parameters():
            param.requires_grad = False

        # Create separate probe layers for each BERT layer
        self.probes = nn.ModuleDict({
            f'layer_{layer}': nn.Linear(hidden_size, num_classes)
            for layer in self.layers_to_probe
        })

        # Final layer to combine predictions from all layers
        self.combination_layer = nn.Linear(
            len(self.layers_to_probe) * num_classes,
            num_classes
        )

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            position_ids: Optional[torch.Tensor] = None,
            head_mask: Optional[torch.Tensor] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            output_attentions: Optional[bool] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        # Run BERT with no_grad to prevent gradient flow
        with torch.no_grad():

            bert_outputs = super(BertProbe, self).forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                output_hidden_states=True,
                return_dict=True
            )

            # Detach hidden states to ensure no gradient flow back to BERT
            hidden_states = tuple(h.detach() for h in bert_outputs.hidden_states)
            bert_logits = bert_outputs.logits.detach()

        layer_logits = {}
        all_layer_predictions = []

        # Process each layer
        for layer_idx in self.layers_to_probe:
            layer_hidden = hidden_states[layer_idx] # [B, Seq, Hidden]
            # Get predictions for this layer
            layer_probe = self.probes[f'layer_{layer_idx}']
            layer_output = layer_probe(layer_hidden) # [B, Seq, num_classes]

            layer_logits[f'layer_{layer_idx}'] = layer_output
            all_layer_predictions.append(layer_output)

        # Combine predictions from all layers
        combined_predictions = torch.cat(all_layer_predictions, dim=-1)
        all_layer_logits = self.combination_layer(combined_predictions)
        return all_layer_logits, bert_logits, layer_logits

    def train(self, mode: bool = True):
        """
        Override train method to ensure BERT stays in eval mode
        """
        super().train(mode)
        self.bert.eval()
        return self
