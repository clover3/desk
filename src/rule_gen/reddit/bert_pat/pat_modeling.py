from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn as nn
from torch.nn import CrossEntropyLoss
from transformers import BertForSequenceClassification, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn.functional as F


@dataclass
class PATOutput(SequenceClassifierOutput):
    logits1: torch.Tensor = None
    logits2: torch.Tensor = None


class CombineByLogitAdd(nn.Module):
    def forward(self, local_decision_a, local_decision_b):
        combine_logits = local_decision_a + local_decision_b
        output = F.softmax(combine_logits, dim=1)
        return output


class CombineByScoreAdd(nn.Module):
    def forward(self, local_decision_a, local_decision_b):
        output = local_decision_a + local_decision_b
        return output


class BertPAT(BertForSequenceClassification):
    def __init__(
            self,
            config,
            combine_layer_factory,
    ):
        super().__init__(config)
        self.combine_layer = combine_layer_factory()

    def forward(
            self,
            input_ids1: torch.Tensor,
            attention_mask1: torch.Tensor,
            input_ids2: torch.Tensor,
            attention_mask2: torch.Tensor,
            token_type_ids: torch.Tensor = None,
            labels: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ) -> PATOutput:

        def apply_bert_cls(input_ids, attention_mask):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

        logits1 = apply_bert_cls(input_ids1, attention_mask1)
        logits2 = apply_bert_cls(input_ids2, attention_mask2)

        logits = self.combine_layer(logits1, logits2)
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output

        return PATOutput(
            loss=loss,
            logits=logits,
            logits1=logits1,
            logits2=logits2,
        )


class BertPatFirst(BertForSequenceClassification):
    def __init__(
            self,
            config,
            combine_layer_factory,
    ):
        super().__init__(config)
        self.combine_layer = combine_layer_factory()

    def forward(
            self,
            input_ids1: torch.Tensor,
            attention_mask1: torch.Tensor,
            return_dict: Optional[bool] = None,
    ) -> PATOutput:

        def apply_bert_cls(input_ids, attention_mask):
            outputs = self.bert(
                input_ids,
                attention_mask=attention_mask,
            )
            pooled_output = outputs[1]
            pooled_output = self.dropout(pooled_output)
            logits = self.classifier(pooled_output)
            return logits

        logits1 = apply_bert_cls(input_ids1, attention_mask1)
        return logits1