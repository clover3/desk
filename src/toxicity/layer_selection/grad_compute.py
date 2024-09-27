import json
import logging
import sys
from collections import defaultdict
from functools import partial

from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import os


import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from chair.list_lib import right, left
from toxicity.cpath import output_root_path
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted, apply_llama_guard_formats, ToxigenTrain, \
    ToxigenBinary
from toxicity.dataset_helper.csv_datasets import load_toxigen_like_csv

pf_log = logging.getLogger(__name__)

def init_logging():
    format_str = '%(levelname)s\t%(name)s \t%(asctime)s %(message)s'
    formatter = logging.Formatter(format_str,
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(formatter)
    root_logger = logging.getLogger()
    root_logger.addHandler(ch)


def generate_label(batch: List[Tuple[str, str]], tokenizer, device):
    txt, tgt = zip(*batch)
    inputs = tokenizer(list(txt), return_tensors="pt", padding=True).to(device)
    inputs_targets = [txt_ + tgt_ for txt_, tgt_ in zip(txt, tgt)]
    inputs_targets = tokenizer(inputs_targets, return_tensors="pt", padding=True).to(device)
    num_prompt_toks = [int((i != tokenizer.pad_token_id).sum()) for i in inputs['input_ids']]
    num_pad_toks = [int((i == tokenizer.pad_token_id).sum()) for i in inputs_targets['input_ids']]
    prompt_len = [x + y for x, y in zip(num_pad_toks, num_prompt_toks)]
    prompt_target_len = inputs_targets['input_ids'].size(1)
    label_mask = torch.tensor([[False] * length + [True] * (prompt_target_len - length) for length in prompt_len]).to(device)
    return inputs_targets, label_mask

def get_loss(logits, inputs_targets, label_mask):
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = inputs_targets['input_ids'][..., 1:].contiguous()
    loss_fct = CrossEntropyLoss(reduction='none')
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    loss = loss.view(inputs_targets['input_ids'].size(0), -1)
    loss = (loss * label_mask[:, 1:]).sum(1) / label_mask[:, 1:].sum(1)
    loss = loss.mean()
    return loss


def filter_params(params, layer_no):
    prefix = f"model.layers.{layer_no}."
    output = []
    for n, p in params:
        if prefix in n:
            output.append((n, p))
    return output


def compute_gradient_abs(model, loader, layer):
    metrics = defaultdict(list)
    average_metrics = defaultdict(float)
    model.train()
    params = filter_params(model.named_parameters(), layer)
    pf_log.info("Computing gradients for: %s", str(left(params)))

    def get_grad_norms(grads):
        _metrics = defaultdict(list)
        for (name, param), grad in zip(params, grads):
            _metrics[name] = torch.norm(grad).item() / torch.norm(param).item()
        return _metrics

    for inputs_targets, label_mask in tqdm(loader):
        outputs = model(**inputs_targets)
        logits = outputs.logits
        loss_xent = get_loss(logits, inputs_targets, label_mask)
        grad_xent = torch.autograd.grad(
            outputs=loss_xent, inputs=right(params), retain_graph=True
        )
        xent_grad_metrics = get_grad_norms(grad_xent)
        for k, v in xent_grad_metrics.items():
            metrics[k].append(v)
    for k, v in metrics.items():
        average_metrics[k] = np.array(v).mean(0)
    return average_metrics


def main():
    init_logging()
    pf_log.setLevel(logging.INFO)
    pf_log.info(__name__)
    model_id = "meta-llama/Meta-Llama-Guard-2-8B"
    pf_log.info(f"Loading toxigen")
    ds = load_toxigen_like_csv("toxigen_train_head_100")
    pf_log.info(f"Applying template")
    edit_payload: List[Tuple[str, str]] = apply_llama_guard_formats(ds)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    tokenizer.pad_token = tokenizer.eos_token
    bs = 1

    pf_log.info(f"Loading models")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pf_log.info(f"Loading Initializing dataloader")
    generate_label_partial = partial(generate_label, tokenizer=tokenizer, device=device)
    loader = DataLoader(edit_payload, batch_size=bs, shuffle=True, collate_fn=generate_label_partial)

    pf_log.info(f"Computing gradients")
    for layer in range(1, 32):
        lr_weights = compute_gradient_abs(model, loader, layer)
        print("Learning rate weights:", lr_weights)
        save_name = f"lg2_{layer}"
        save_path: str = os.path.join(output_root_path, "grad_mag", f"{save_name}.csv")
        json.dump(lr_weights, open(save_path, "w"))


if __name__ == "__main__":
    main()