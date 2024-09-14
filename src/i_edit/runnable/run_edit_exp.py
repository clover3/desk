import fire
import torch
import json
import logging
import os
from transformers import AutoTokenizer, AutoModelForCausalLM

from toxicity.misc_lib import average

LOG = logging.getLogger(__name__)


def load_test_data():
    zsre_file_name = "benchmark_ZsRE_ZsRE-test-all.json"
    file_path = os.path.join("data", zsre_file_name)
    return json.load(open(file_path, "r"))


def compute_sequence_probability(model, tokenizer, prompt, target):
    concat_prompt = prompt + " " + target
    enc_prompt = tokenizer.encode(prompt)
    prompt_length = len(enc_prompt)
    enc_concat = tokenizer.encode(concat_prompt, return_tensors="pt").to("cuda")
    enc_target = tokenizer.encode(target, return_tensors="pt")

    with torch.no_grad():
        outputs = model(enc_concat)
        logits = outputs.logits

    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)

    sequence_log_prob = 0
    for i in range(prompt_length - 1, enc_concat.shape[1] - 1):
        token_log_prob = log_probs[0, i, enc_concat[0, i + 1]]
        sequence_log_prob += token_log_prob

    sequence_prob = torch.exp(sequence_log_prob).item()
    return sequence_prob


def main(model_name="meta-llama/Meta-Llama-3-8B"):
    print("main")
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    all_data = load_test_data()
    model_id = model_name
    print("Model", model_id)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id)
    device = torch.device(f'cuda:0')
    model = model.to(device)

    def get_prob(prompt, target):
        return compute_sequence_probability(model, tokenizer, prompt, target)

    # for each item
    #  Generate prompt
    suc_list = []
    for e in all_data[:100]:
        question_prompt = e['prompt']
        old_gold = e['ground_truth'][0]
        new_target = e['target_new']
        if old_gold == new_target:
            continue
        new_prompt = f"Imagine that the answer to {question_prompt} is {new_target}. {question_prompt}"
        score_old = get_prob(new_prompt, old_gold)
        score_new = get_prob(new_prompt, new_target)
        edit_suc = score_new > score_old
        print(edit_suc, score_old, score_new, question_prompt, new_target)
        suc_list.append(int(edit_suc))

    print(average(suc_list))


if __name__ == "__main__":
    fire.Fire(main)