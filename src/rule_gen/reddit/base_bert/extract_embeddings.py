import fire
import numpy as np
import logging
import os
import pickle
import sys
from typing import List, Tuple, Union, Iterator

import torch
from tqdm import tqdm
from transformers import BertModel, AutoTokenizer
from transformers import BertTokenizer, BertForSequenceClassification

from chair.list_lib import left
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.classifier_loader.torch_misc import get_device
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex

LOG = logging.getLogger(__name__)

def setup_bert_model(model_name: str, num_labels: int):
    """
    Set up BERT model with output_hidden_states enabled
    """
    model = BertForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        output_hidden_states=True  # Enable hidden states output
    )
    return model




class BertHiddenStatesExtractor:
    def __init__(self, run_name: str):
        self.model_path = get_model_save_path(run_name)
        LOG.info(f"Loading BERT model from: {self.model_path}")
        self.model = BertModel.from_pretrained(self.model_path, output_hidden_states=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.device = get_device()
        self.model.to(self.device)
        self.model.eval()
        self.batch_size = 16
        self.max_length = 512

    def get_pooled_output(
            self,
            text: Union[str, List[str]]
    ) -> torch.Tensor:
        is_single = isinstance(text, str)
        if is_single:
            text = [text]

        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=512
        )

        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)

        # Get pooled output
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            pooled_output = outputs.pooler_output  # (batch_size, hidden_size)

        if is_single:
            pooled_output = pooled_output[0:1, :]  # Keep batch dimension but set to 1

        return pooled_output

    def get_hidden_states(
            self,
            text: Union[str],
            layers: List[int] = [-4, -3, -2, -1]  # Last 4 layers by default
    ) -> torch.Tensor:
        LOG.info("get_hidden_states for %d items", len(text))
        text = [text]
        # Tokenize
        encoded = self.tokenizer(
            text,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )
        return self._get_hidden_states(encoded, layers)


    def _get_hidden_states(self, encoded, layers):
        # Move to device
        input_ids = encoded['input_ids'].to(self.device)
        attention_mask = encoded['attention_mask'].to(self.device)
        # Get hidden states
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
        # Extract hidden states from specified layers
        all_hidden_states = outputs.hidden_states  # tuple of tensors
        selected_hidden_states = []
        for layer_idx in layers:
            # Get layer's hidden states
            layer_hidden_states = all_hidden_states[layer_idx]
            selected_hidden_states.append(layer_hidden_states)
        stacked_hidden_states = torch.stack(selected_hidden_states, dim=1)
        return stacked_hidden_states

    def _batch_texts(self, texts: List[str]) -> Iterator[List[str]]:
        """Helper method to create batches of texts"""
        for i in range(0, len(texts), self.batch_size):
            yield texts[i:i + self.batch_size]


    def get_sentence_embedding(
            self,
            text: Union[str, List[str]],
            layers: List[int] = [-4, -3, -2, -1],
            pooling_strategy: str = 'cls'
    ) -> np.array:
        hidden_states, attention_mask = self.get_hidden_states(text, layers)
        # hidden_states shape: (num_layers, batch_size, seq_len, hidden_size)

        if pooling_strategy == 'cls':
            # Use [CLS] token representation from each layer
            sentence_embeddings = hidden_states[:, :, 0, :]  # (num_layers, batch_size, hidden_size)
        elif pooling_strategy == 'mean':
            # Mean pooling for each layer
            mask = attention_mask.unsqueeze(0).unsqueeze(-1)  # (1, batch_size, seq_len, 1)
            sentence_embeddings = (hidden_states * mask).sum(dim=2) / mask.sum(
                dim=2)  # (num_layers, batch_size, hidden_size)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling_strategy}")

        return sentence_embeddings.cpu().numpy()

    def get_sentence_embedding_batched(self,
                                       texts: List[str],
                                       layers: List[int] = [-4, -3, -2, -1],
                                       pooling_strategy: str = 'cls'
                                       ):
        emb_list = []
        show_progress = len(texts) > 50
        iterator = tqdm(self._batch_texts(texts)) if show_progress else self._batch_texts(texts)
        for b in iterator:
            emb = self.get_sentence_embedding(b, layers, pooling_strategy)
            emb_list.append(emb)

        emb_all = np.concatenate(emb_list, axis=0)
        return emb_all

    def get_sub_text_embeddings(self, full_text, sub_text_list, layer_no=-1):
        text_sp_rev = " ".join(full_text.split())
        sub_text_embeddings = []

        # Get full text embeddings
        encoded = self.tokenizer(
            text_sp_rev,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=self.max_length
        )
        full_text_embeddings = self._get_hidden_states(
            encoded,
            layers=[layer_no])

        for sub_text in sub_text_list:
            # Get the sub-text for this window
            # Get token positions for this sub-text in the original text
            sub_text_start_char = text_sp_rev.find(sub_text)
            sub_text_end_char = sub_text_start_char + len(sub_text) - 1

            st = encoded.char_to_token(0, sub_text_start_char)
            ed = encoded.char_to_token(0, sub_text_end_char) + 1

            sub_emb = full_text_embeddings[0, 0, st: ed, :]
            sub_emb = sub_emb.cpu().numpy()
            sub_text_embeddings.append(sub_emb)

        return sub_text_embeddings


def main(sb = "history"):
    extractor = BertHiddenStatesExtractor(f"bert2_{sb}")
    save_path = get_reddit_train_data_path_ex("train_data2", sb, "train")
    items = read_csv(save_path)
    n_item = 10000
    texts = left(items[:n_item])

    batch_embeddings = extractor.get_sentence_embedding_batched(texts)
    print(f"Batch embeddings shape: {batch_embeddings.shape}")

    save_path = os.path.join(output_root_path, "reddit", "pickles", f"bert2_{sb}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(batch_embeddings, f)


if __name__ == "__main__":
    fire.Fire(main)