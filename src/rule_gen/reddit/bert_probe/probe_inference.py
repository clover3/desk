import os
from dataclasses import dataclass
from typing import List

import fire
import numpy as np
import scipy.special
import torch
from transformers import AutoTokenizer

from chair.html_visual import HtmlVisualizer, Cell
from chair.list_lib import right
from chair.misc_lib import path_join
from chair.tokens_util import cells_from_tokens
from desk_util.path_helper import get_model_save_path, load_csv_dataset
from desk_util.runnable.run_eval import load_labels
from rule_gen.cpath import output_root_path
from rule_gen.reddit.bert_probe.probe_model import BertProbe

os.environ["TOKENIZERS_PARALLELISM"] = "true"


@dataclass
class ProbeOutput:
    all_layer_logits: np.ndarray  # Shape: (num_layers, batch_size, seq_len, num_classes)
    bert_logits: np.ndarray  # Shape: (batch_size, num_classes)
    layer_logits: dict[str, np.ndarray]  # Shape: (batch_size, seq_len, num_classes)


class ProbeInference:
    def __init__(self, model_path: str, device: str = None, max_length=256):
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
        base_model = 'bert-base-uncased'
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.model = BertProbe.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
        self.max_length = max_length

    def predict(self, text: str) -> ProbeOutput:
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs.to(self.device)
        with torch.no_grad():
            all_layer_logits, bert_logits, layer_logits = self.model(**inputs)
            return ProbeOutput(
                all_layer_logits=all_layer_logits.cpu().numpy(),
                bert_logits=bert_logits.cpu().numpy(),
                layer_logits={k: v.cpu().numpy() for k, v in layer_logits.items()},
            )

    def predict_get_layer_mean(self, text):
        inputs = self.tokenizer(
            text,
            padding='max_length',
            truncation=True,
            max_length=self.max_length,
            return_tensors='pt'
        )
        inputs.to(self.device)
        with torch.no_grad():
            all_layer_logits, bert_logits, layer_logits = self.model(**inputs)
        probe_list = []
        for layer_no in range(len(layer_logits)):
            layer_logit = layer_logits[f"layer_{layer_no}"]
            probe_list.append(layer_logit)

        probe_concat = torch.stack(probe_list, dim=2)
        probes = torch.softmax(probe_concat, dim=-1)
        probe = torch.mean(probes, dim=2)
        return probe.cpu().numpy()


    def predict_cls_label(self, str):
        output = self.predict(str)
        preds = np.argmax(output.bert_logits, axis=-1)
        return preds[0]


# Example usage:
def main(sb="askscience", model_name=""):
    if not model_name:
        model_name = f"bert2_{sb}"
        model_name = model_name + "_probe"
    model_path = get_model_save_path(model_name)
    inf = ProbeInference(model_path)
    dataset = f"{sb}_2_val_100"
    data = load_csv_dataset(dataset)
    labels = load_labels(dataset)
    # outputs = [inf.predict(t) for t in right(data)]
    save_path = path_join(output_root_path, "visualize", f"{dataset}_probe.html")
    html = HtmlVisualizer(save_path)

    def layer_no_to_name(layer_no):
        if layer_no == 0:
            return "embed"
        else:
            return "layer_{}".format(layer_no - 1)

    for t, label in zip(right(data)[:100], labels):
        item = inf.predict(t)
        html.write_paragraph("Text: {}".format(t))
        inputs = inf.tokenizer(t)
        tokens = inf.tokenizer.convert_ids_to_tokens(inputs["input_ids"])
        n_tokens = min(len(tokens), 256)
        text_rows = [Cell("")] + cells_from_tokens(tokens)
        rows = [text_rows]
        cls_probs = scipy.special.softmax(item.bert_logits, axis=-1)[0]
        html.write_paragraph("Prediction: {}".format(cls_probs))

        for layer_no in range(len(item.layer_logits)):
            layer_logit = item.layer_logits[f"layer_{layer_no}"]
            probs = scipy.special.softmax(layer_logit, axis=-1)[0]  # [B,
            def prob_to_one_digit(p):
                v = int(p * 10 + 0.05)
                if v > 9:
                    return "A"
                else:
                    s = str(v)
                    assert len(s) == 1
                    return s

            row = [Cell(layer_no_to_name(layer_no))]
            for seq_idx in range(n_tokens):
                case_probs = probs[seq_idx]
                prob_digits: List[str] = list(map(prob_to_one_digit, case_probs))
                cell_str = "".join(prob_digits)
                # ret = highlight_term(layer_no, seq_idx)
                # if ret:
                #     cell_str += ret
                color_score = [
                    1-case_probs[1],  # R
                    1-case_probs[1],  # G
                    1,  # B
                ]
                color_score = [255 * t for t in color_score]
                color = "".join([("%02x" % int(v)) for v in color_score])
                cell = Cell(cell_str, 255, target_color=color)
                row.append(cell)
            rows.append(row)
        html.write_table(rows)


if __name__ == "__main__":
    fire.Fire(main)
