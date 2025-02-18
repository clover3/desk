import torch
import logging

import fire
from transformers import AutoTokenizer

from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.bert_pat.partition_util import get_non_sharp_indices
from rule_gen.reddit.bert_pat.pat_modeling import BertPAT, CombineByScoreAdd
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger("InfPat")


class PatInference:
    def __init__(self, sb="TwoXChromosomes", debug=False, window_size=1):
        self.sb = sb
        self.debug = debug
        self.model_name = f"bert_ts_{sb}"
        self.max_length = 256
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._init_model()
        self.window_size = window_size

    def _init_model(self):
        """Initialize the model and tokenizer"""
        with JobContext(self.model_name + "_train"):
            model_path = get_model_save_path(self.model_name)
            LOG.info("Loading PAT model from %s", model_path)
            self.tokenizer = AutoTokenizer.from_pretrained(model_path)
            self.model = BertPAT.from_pretrained(
                model_path,
                combine_layer_factory=CombineByScoreAdd
            )
            self.model.to(self.device)


    def get_first_view_scores(self, text, window_size=None):
        if window_size is None:
            window_size = self.window_size
        tokens = self.tokenizer.tokenize(text)
        candidate_indices = get_non_sharp_indices(tokens)
        ret = []
        for i in range(len(candidate_indices) - window_size):
            start_idx = candidate_indices[i]
            end_idx = candidate_indices[i + window_size]
            sub_seq: list[str] = tokens[start_idx:end_idx]
            sub_text = self.tokenizer.convert_tokens_to_string(sub_seq)
            encoded, inputs = self.encode_input(sub_seq)
            with torch.no_grad():
                outputs = self.model(**inputs, return_dict=True)

            probs1 = torch.softmax(outputs.logits1, -1)
            result = {
                'sequence': sub_text,
                'input_ids': encoded['input_ids'][0].cpu().numpy(),
                'score': probs1[0, 1].cpu().numpy(),
            }
            ret.append(result)
        return ret

    def encode_input(self, sub_seq):
        encoded = self.tokenizer(
            sub_seq,
            is_split_into_words=True,  # Add this flag to indicate pre-tokenized input
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )
        inputs = {
            'input_ids1': encoded['input_ids'].to(self.device),
            'attention_mask1': encoded['attention_mask'].to(self.device),
            'input_ids2': encoded['input_ids'].to(self.device),
            'attention_mask2': encoded['attention_mask'].to(self.device),
            'labels': torch.tensor([1], device=self.device)
        }
        return encoded, inputs


def infer_tokens(sb="TwoXChromosomes"):
    init_logging()
    model_name = f"bert_ts_{sb}"
    data_name = "train_data2"
    LOG.setLevel(logging.INFO)

    t = 0
    with JobContext(model_name + "_train"):
        train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
            data_name, sb, "train"))

        pat = PatInference(sb=sb, window_size=5)
        for example in train_dataset:
            text = example['text']
            window_size = 5
            max_score = 0
            best_seq = None
            LOG.info("Text: %s", text)
            while max_score == 0:
                ret = pat.get_first_view_scores(text, window_size=window_size)
                LOG.debug("Trying window size %d", window_size)
                for result in ret:
                    if result["score"] > t:
                        if result["score"] > max_score:
                            max_score = result["score"]
                            best_seq = result["sequence"]

                window_size += 1
                s_outs = ["{0} ({1:.2f})".format(result['sequence'], result["score"]) for result in ret]
                LOG.debug(" ".join(s_outs))
            LOG.info("Best seq: {} ({})".format(best_seq, round(float(max_score), 2)))


if __name__ == "__main__":
    fire.Fire(infer_tokens)
