from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers import BertPreTrainedModel, AutoConfig, BertConfig, AutoTokenizer, AutoModel
from transformers.modeling_outputs import SequenceClassifierOutput


@dataclass
class C1Config(BertConfig):
    def __init__(
            self,
            base_model_name: str = 'bert-base-uncased',
            n_sb=100,
            **kwargs,
    ):
        self.base_model_name = base_model_name
        self.n_sb = n_sb
        super().__init__(**kwargs)


class BertC1(BertPreTrainedModel):
    def __init__(
            self,
            config,
    ):
        base_config = AutoConfig.from_pretrained(
            config.base_model_name, trust_remote_code=True
        )
        for key, value in vars(base_config).items():
            if not hasattr(config, key):  # Only set if not already in config
                setattr(config, key, value)
        super().__init__(config)
        self.embeddings = nn.Embedding(config.n_sb, config.hidden_size + 1)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.bert = AutoModel.from_pretrained(config.base_model_name)

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            sb_id: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            return_dict=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[1]
        embeddings = self.embeddings(sb_id)
        w = embeddings[:, :-1]  # [B, H]
        b = embeddings[:, -1]  # [B]
        weighted_sum = torch.sum(w * pooled_output, dim=-1)
        logits = weighted_sum + b
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            labels = labels.float().view(-1)
            loss = loss_fct(logits, labels)
        else:
            loss = 0.0
        if not return_dict:
            output = (logits,)
            return ((loss,) + output) if loss is not None else output
        else:
            return SequenceClassifierOutput(
                loss=loss,
                logits=logits,
            )



