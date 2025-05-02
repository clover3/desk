from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn as nn
from torch.nn import BCEWithLogitsLoss
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers import BertPreTrainedModel, AutoConfig, AutoTokenizer, AutoModel, BertConfig


@dataclass
class C2Output(SequenceClassifierOutput):
    p_loss: torch.Tensor = None
    logit_loss: torch.Tensor = None
    policy_pred: torch.Tensor = None
    penalty_losses: torch.Tensor = None
    w: torch.Tensor = None
    b: torch.Tensor = None


@dataclass
class C2Config(BertConfig):
    def __init__(
            self,
            alpha: float = 0.1,
            base_model_name: str = 'bert-base-uncased',
            n_sb: int = 100,
            n_policy: int = 72,
            n_gen_policy: int = 8,
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.base_model_name = base_model_name
        self.n_sb = n_sb
        self.n_policy = n_policy
        self.n_gen_policy = n_gen_policy

class BertC2(BertPreTrainedModel):
    def __init__(
            self,
            config,
    ):
        base_config = AutoConfig.from_pretrained(
            config.base_model_name, trust_remote_code=True
        )
        for key, value in vars(base_config).items():
            setattr(config, key, value)

        super().__init__(config)
        self.project1 = torch.nn.Linear(base_config.hidden_size, config.n_policy)
        self.embeddings = nn.Embedding(config.n_sb, config.n_policy + 1)
        self.tokenizer = AutoTokenizer.from_pretrained(config.base_model_name)
        self.model = AutoModel.from_pretrained(config.base_model_name)
        self.n_gen_policy = config.n_gen_policy
        self.n_policy = config.n_policy
        self.alpha = config.alpha
        policy_penalty_mask = [0] * self.n_gen_policy + [1] * (self.n_policy - self.n_gen_policy)
        m = torch.Tensor(policy_penalty_mask)
        self.register_buffer('policy_penalty_mask', m.unsqueeze(0))

    def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            sb_id: torch.Tensor,
            token_type_ids: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            policy_labels: Optional[torch.Tensor] = None,
            policy_label_mask: Optional[torch.Tensor] = None,
            return_dict: Optional[bool] = None,
    ):
        outputs = self.model(
            input_ids,
            attention_mask=attention_mask,
        )
        pooled_output = outputs[1]
        policy_pred = self.project1(pooled_output)  # [B, M]
        if policy_labels is not None:
            loss_fct = BCEWithLogitsLoss(reduction="none")
            p_loss = loss_fct(policy_pred, policy_labels.float())  # [B, M]
            s = p_loss * policy_label_mask.float()  # [B, M]
            p_loss = torch.sum(s, dim=1)   # [B]
            p_loss = torch.mean(p_loss, dim=0)
        else:
            p_loss = 0.0

        embeddings = self.embeddings(sb_id)
        w = embeddings[:, :-1]  # [B, H]
        b = embeddings[:, -1]  # [B]
        final_logits = torch.sum(w * policy_pred, dim=1) + b
        if labels is not None:
            loss_fct = BCEWithLogitsLoss()
            logit_loss = loss_fct(final_logits, labels.float())
        else:
            logit_loss = 0.0

        penalty_losses = torch.sum(torch.abs(policy_pred) * self.policy_penalty_mask, dim=1)
        penalty_losses = torch.mean(penalty_losses)

        loss = logit_loss + p_loss + penalty_losses * self.alpha
        if not return_dict:
            output = (final_logits,)
            return ((loss,) + output) if loss is not None else output
        else:
            return C2Output(
                loss=loss if labels is not None else None,
                logits=final_logits,
                policy_pred=policy_pred,
                logit_loss=logit_loss,
                p_loss=p_loss,
                penalty_losses=penalty_losses,
                w=w,
                b=b,
            )
