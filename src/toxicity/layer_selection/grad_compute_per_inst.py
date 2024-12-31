import json
import logging
import sys
from functools import partial

from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
import os

import torch
from typing import List, Tuple

from rule_gen.cpath import output_root_path
from toxicity.dataset_helper.load_toxigen import apply_llama_guard_formats
from desk_util.csv_datasets import load_toxigen_like_csv
from toxicity.layer_selection.grad_compute import compute_gradient_abs, generate_label

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
        for item_idx, item in enumerate(loader):
            lr_weights = compute_gradient_abs(model, [item], layer)
            print("Learning rate weights:", lr_weights)
            save_name = f"lg2_{item_idx}_{layer}"
            save_path: str = os.path.join(output_root_path, "grad_mag_inst", f"{save_name}.json")
            json.dump(lr_weights, open(save_path, "w"))


if __name__ == "__main__":
    main()