from abc import abstractmethod
from typing import Optional, Union, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, BCELoss
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
from transformers.modeling_outputs import SequenceClassifierOutput
import string


# forward -> get_score -> score

class ColA(BertPreTrainedModel):
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
        mask_punctuation = False
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
        scores = self.compute_inter_scores(Q, D, D_mask)
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

    def compute_inter_scores(self, Q, D_padded, D_mask):
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
        scores = self.compute_interaction_score(Q, D, D_mask)
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

    def compute_interaction_score(self, Q, D_padded, D_mask):
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


class ColC(ColA):
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


# forward -> get_score -> score
class ColD(ColC):
    def compute_inter_scores(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores = scores.max(dim=-1).values

        def reduce_fn(t):
            return t.max(dim=-1).values

        scores = reduce_fn(scores)
        return scores


class ColE(ColC):
    def compute_inter_scores(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores = scores.max(dim=-1).values

        def reduce_fn(t):
            weights = torch.softmax(t, dim=-1)
            return torch.sum(weights * t, dim=-1)

        scores = reduce_fn(scores)
        return scores


class ColF(ColC):
    def compute_inter_scores(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)

        def reduce_fn(t):
            weights = torch.softmax(t, dim=-1)
            return torch.sum(weights * t, dim=-1)

        scores_1d = reduce_fn(scores_2d)
        scores = reduce_fn(scores_1d)
        return scores


class ColG(ColC):
    def compute_inter_scores(self, Q, D_padded, D_mask):
        scores = torch.matmul(Q, D_padded.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        probs_2d = torch.sigmoid(scores_2d)

        def reduce_fn(t):
            weights = torch.softmax(t, dim=-1)
            return torch.sum(weights * t, dim=-1)

        scores_1d = reduce_fn(probs_2d)
        scores = reduce_fn(scores_1d)
        return scores

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

        probs = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        probs = probs.unsqueeze(1)
        print("probs.shape", probs.shape)
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


class ColH(ColG):
    def __init__(self, config, ):
        super(ColH, self).__init__(config)
        self.linear_mask = nn.Linear(self.bert.config.hidden_size, 2)

    @abstractmethod
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)

        Q_mask_logits = self.linear_mask(Q_enc)
        Q_dyn_mask = torch.softmax(Q_mask_logits, dim=2)[:, :, 1]

        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()
        Q = Q * mask

        Q = F.normalize(Q, p=2, dim=2)

        return Q, Q_dyn_mask

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q, Q_dyn_mask = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')

        scores = torch.matmul(Q, D.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        probs_2d = torch.sigmoid(scores_2d)

        def reduce_fn(t):
            weights = torch.softmax(t, dim=-1)
            return torch.sum(weights * t, dim=-1)

        scores_1d = reduce_fn(probs_2d)
        scores_1d = scores_1d * Q_dyn_mask
        scores = reduce_fn(scores_1d)

        return scores


class ColI(ColC):
    def __init__(self, config, ):
        super(ColI, self).__init__(config)
        self.linear_mask = nn.Linear(self.bert.config.hidden_size, 2)

    @abstractmethod
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)
        Q = Q * mask
        Q = F.normalize(Q, p=2, dim=2)

        Q_mask_logits = self.linear_mask(Q_enc)
        Q_dyn_mask = torch.softmax(Q_mask_logits, dim=2)
        Q_dyn_mask = Q_dyn_mask * mask
        Q_dyn_mask = Q_dyn_mask[:, :, 1]

        return Q, Q_dyn_mask

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q, Q_dyn_mask = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')

        scores = torch.matmul(Q, D.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)

        def reduce_fn(t):
            weights = torch.softmax(t, dim=-1)
            return torch.sum(weights * t, dim=-1)

        scores_1d = reduce_fn(scores_2d)
        scores_1d = scores_1d + (Q_dyn_mask * -0.2)
        scores = reduce_fn(scores_1d)
        return scores

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

        scores = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        scores = scores.unsqueeze(1)
        probs = torch.sigmoid(scores)
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


class ColJ(ColC):
    def __init__(self, config, ):
        super(ColJ, self).__init__(config)
        self.linear_qk = nn.Linear(self.bert.config.hidden_size, self.dim)
        self.linear_q_w = nn.Linear(self.bert.config.hidden_size, 1)

    @abstractmethod
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)
        Q = Q * mask
        Q = F.normalize(Q, p=2, dim=2)

        Q_q = self.linear_qk(Q_enc)
        Q_w = self.linear_q_w(Q_enc)
        return Q, Q_q, Q_w

    @abstractmethod
    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D_enc)

        mask = torch.tensor(
            self.mask(input_ids, skiplist=self.skiplist),
            device=self.device
        ).unsqueeze(2).float()
        D = D * mask
        D = F.normalize(D, p=2, dim=2)
        D_q = self.linear_qk(D_enc)


        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, D_q, mask.bool()

        return D, D_q
    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q, Q_q, Q_w = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_q, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')

        scores = torch.matmul(Q, D.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)

        weights_raw = torch.matmul(Q_q, D_q.transpose(-2, -1))
        weights_raw = weights_raw.masked_fill(D_padding.unsqueeze(1), -9999)
        weights = torch.softmax(weights_raw, dim=-1)
        per_q_scores = torch.sum(scores_2d * weights, dim=-1)

        q_padding = ~(query_attention_mask.bool())
        Q_w = Q_w.squeeze(2)
        Q_w = Q_w.masked_fill(q_padding, -9999)
        q_weights = torch.softmax(Q_w, dim=-1)
        return torch.sum(per_q_scores * q_weights, dim=-1)

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
        scores = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        scores = scores.unsqueeze(1)
        probs = torch.sigmoid(scores)
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


class ColK(ColC):
    @abstractmethod
    def query(self, input_ids, attention_mask):
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)
        Q = Q * mask
        Q = F.normalize(Q, p=2, dim=2)
        return Q

    @abstractmethod
    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']

        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D_enc)

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

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        D_mask, scores = self.get_2d_scores(query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask)
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores_1d = torch.max(scores_2d, dim=-1).values
        score = torch.max(scores_1d, dim=-1).values
        return score

    def get_2d_scores(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        Q = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')
        scores = torch.matmul(Q, D.transpose(-2, -1))
        return scores


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
        scores = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        scores = scores.unsqueeze(1)
        probs = torch.sigmoid(scores)
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


def insert_mark(input_ids, MARK_ID):
    head = input_ids[:, :1]
    tail = input_ids[:, 1:-1]
    mark = torch.ones([input_ids.size(0), 1], dtype=torch.int64) * MARK_ID
    mark = mark.to(head.device)
    input_ids = torch.concat([head, mark, tail], dim=1)
    return input_ids

class ColL(ColA):
    @abstractmethod
    def query(self, input_ids, attention_mask):
        Q_MARK_ID = 80
        input_ids = insert_mark(input_ids, Q_MARK_ID)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)
        Q = Q * mask
        Q = F.normalize(Q, p=2, dim=2)
        return Q

    @abstractmethod
    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']
        D_MARK_ID = 81
        input_ids = insert_mark(input_ids, D_MARK_ID)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D_enc)

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
        scores = torch.matmul(Q, D.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)
        scores_1d = torch.max(scores_2d, dim=-1).values
        score = torch.max(scores_1d, dim=-1).values
        return score

    def get_2d_scores(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        Q = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')
        scores = torch.matmul(Q, D.transpose(-2, -1))
        return scores


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
        scores = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        scores = scores.unsqueeze(1)
        probs = torch.sigmoid(scores)
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


class ColM(ColC):
    def __init__(self, config, ):
        super(ColM, self).__init__(config)
        self.linear_qk = nn.Linear(self.bert.config.hidden_size, self.dim)
        self.linear_q_w = nn.Linear(self.bert.config.hidden_size, 1)

    @abstractmethod
    def query(self, input_ids, attention_mask):
        Q_MARK_ID = 80
        input_ids = insert_mark(input_ids, Q_MARK_ID)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        mask = torch.tensor(
            self.mask(input_ids, skiplist={}),
            device=self.device
        ).unsqueeze(2).float()

        Q_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        Q = self.linear(Q_enc)
        Q = Q * mask
        Q_q = self.linear_qk(Q_enc)
        Q_w = self.linear_q_w(Q_enc)
        return Q, Q_q, Q_w

    @abstractmethod
    def doc(self, input_ids, attention_mask, keep_dims='return_mask'):
        assert keep_dims in [True, False, 'return_mask']
        D_MARK_ID = 81
        input_ids = insert_mark(input_ids, D_MARK_ID)
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        D_enc = self.bert(input_ids, attention_mask=attention_mask).last_hidden_state
        D = self.linear(D_enc)

        mask = torch.tensor(
            self.mask(input_ids, skiplist=self.skiplist),
            device=self.device
        ).unsqueeze(2).float()
        D = D * mask
        D_q = self.linear_qk(D_enc)

        if keep_dims is False:
            D, mask = D.cpu(), mask.bool().cpu().squeeze(-1)
            D = [d[mask[idx]] for idx, d in enumerate(D)]
        elif keep_dims == 'return_mask':
            return D, D_q, mask.bool()

        return D, D_q


    def get_2d_scores(self, query_input_ids, query_attention_mask, doc_input_ids, doc_attention_mask):
        Q, Q_q, Q_w = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_q, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                                  keep_dims='return_mask')

        scores = torch.matmul(Q, D.transpose(-2, -1))
        return scores

    def get_score(
            self,
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
    ):
        Q, Q_q, Q_w = self.query(input_ids=query_input_ids, attention_mask=query_attention_mask)
        D, D_q, D_mask = self.doc(input_ids=doc_input_ids, attention_mask=doc_attention_mask,
                             keep_dims='return_mask')

        scores = torch.matmul(Q, D.transpose(-2, -1))
        D_padding = ~D_mask.view(scores.size(0), scores.size(-1)).bool()
        scores_2d = scores.masked_fill(D_padding.unsqueeze(1), -9999)

        weights_raw = torch.matmul(Q_q, D_q.transpose(-2, -1))
        weights_raw = weights_raw.masked_fill(D_padding.unsqueeze(1), -9999)
        weights = torch.softmax(weights_raw, dim=-1)
        per_q_scores = torch.sum(scores_2d * weights, dim=-1)

        q_padding = ~(query_attention_mask.bool())
        Q_w = Q_w.squeeze(2)
        Q_w = Q_w.masked_fill(q_padding, -9999)
        q_weights = torch.softmax(Q_w, dim=-1)
        return torch.sum(per_q_scores * q_weights, dim=-1)

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
        scores = self.get_score(
            query_input_ids,
            query_attention_mask,
            doc_input_ids,
            doc_attention_mask,
        )
        scores = scores.unsqueeze(1)
        probs = torch.sigmoid(scores)
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


def get_arch_class(arch_name):
    return {
        "col1": ColA,
        "col3": ColC,  # Add token type embedding
        "colD": ColD,  #
        "colE": ColE,  # Use soft-max weighted sum to reduce Q-side
        "colF": ColF,  # + Use soft-max weighted sum for both Q/D
        "colG": ColG,  # + Apply sigmoid before weighted sum
        "colH": ColH,  # + Add dynamic masking based on q_enc
        "colI": ColI,  # ColF + Add dynamic masking
        "colJ": ColJ,  # Attention-like
        "colK": ColK,  # Reduce by max / max
        "colL": ColL,  # Add Special tokens
        "colM": ColM,  # Attention-like, special token, no norm
    }[arch_name]
