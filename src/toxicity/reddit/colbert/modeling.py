from abc import abstractmethod
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import string




class ColBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dim = 768
        self.linear = nn.Linear(self.bert.config.hidden_size, self.dim)

        # Initialize weights and apply final processing
        self.post_init()
        self.tokenizer = None
        self.pad_token = None
        self.skiplist = None

    def colbert_set_up(self, tokenizer):
        self.tokenizer = tokenizer
        mask_punctuation = True
        self.pad_token = self.tokenizer.pad_token_id
        if mask_punctuation:
            self.skiplist = {
                token_id: True
                for symbol in string.punctuation
                for token_id in [
                    symbol,
                    self.tokenizer.encode(symbol, add_special_tokens=False)[0]
                ]
            }
        else:
            self.skiplist = {}

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')
        scores = self.score(Q, D, D_mask)
        return scores

    @abstractmethod
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        Q = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q)

        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()
        Q = Q * mask

        Q = F.normalize(Q, p=2, dim=2)

        return Q

    @abstractmethod
    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D)

        mask = torch.tensor(
            self.mask(input_ids, skiplist=self.skiplist),
            device=self.device
        ).unsqueeze(2).float()
        D = D * mask

        D = F.normalize(D, p=2, dim=2)

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores = scores.max(dim=-1).values
        scores = scores.sum(dim=-1)
        return scores

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            loss_fct = BCELoss()
            # No need to reshape scores since we're treating this as binary classification
            labels = labels.unsqueeze(1)
            loss = loss_fct(probs, labels.float())
        if not return_dict:
            output = probs
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=probs,
        )





class Col2(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        self.dim = 768
        self.linear = nn.Linear(self.bert.config.hidden_size, self.dim)

        # Initialize weights and apply final processing
        self.post_init()
        self.tokenizer = None
        self.pad_token = None
        self.skiplist = None
        self.loss_option = "bce"

    def colbert_set_up(self, tokenizer):
        self.tokenizer = tokenizer
        mask_punctuation = True
        self.pad_token = self.tokenizer.pad_token_id
        if mask_punctuation:
            self.skiplist = {
                token_id: True
                for symbol in string.punctuation
                for token_id in [
                    symbol,
                    self.tokenizer.encode(symbol, add_special_tokens=False)[0]
                ]
            }
        else:
            self.skiplist = {}

    def mask(self, input_ids, skiplist):
        mask = [
            [(x not in skiplist) and (x != self.pad_token) for x in d]
            for d in input_ids.cpu().tolist()
        ]
        return mask

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')
        scores = self.score(Q, D, D_mask)
        return scores


    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        Q = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q)

        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()
        Q = Q * mask
        return Q

    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D)

        mask = torch.tensor(
            self.mask(input_ids, skiplist=self.skiplist),
            device=self.device
        ).unsqueeze(2).float()
        D = D * mask

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D

    def score(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores = scores.max(dim=-1).values
        scores = scores.sum(dim=-1)
        return scores

    @property
    def device(self):
        return next(self.parameters()).device

    def forward(
        self,
        query_input_ids: Optional[torch.Tensor] = None,
        query_attention_mask: Optional[torch.Tensor] = None,
        doc_input_ids: Optional[torch.Tensor] = None,
        doc_attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        logits = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        logits = logits.unsqueeze(1)
        probs = torch.sigmoid(logits)
        loss = None
        if labels is not None:
            if self.loss_option == "bce": # bce
                loss_fct = BCELoss()
                # No need to reshape scores since we're treating this as binary classification
                labels = labels.unsqueeze(1)
                loss = loss_fct(probs, labels.float())
            else:
                raise ValueError()

        if not return_dict:
            output = probs
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=probs,
        )


class Col3(ColBertForSequenceClassification):
    # Add Token_Type ID
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.zeros_like(input_ids, device=self.device)

        Q = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q)
        Q = F.normalize(Q, p=2, dim=2)

        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()
        Q = Q * mask
        return Q

    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        token_type_ids = torch.ones_like(input_ids, device=self.device)

        D = self.bert(
            input_ids,
            token_type_ids=token_type_ids,
            attention_mask=attention_mask).last_hidden_state
        D = self.linear(D)
        D = F.normalize(D, p=2, dim=2)


        mask = torch.tensor(
            self.mask(input_ids, skiplist=self.skiplist),
            device=self.device
        ).unsqueeze(2).float()
        D = D * mask

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, mask.bool()

        return D