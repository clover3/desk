# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# from accelerate import init_empty_weights, load_checkpoint_and_dispatch

import fire
import os
import sys
import time

import torch
from transformers import AutoTokenizer
from transformers import AutoModelForCausalLM, LlamaForCausalLM, LlamaConfig, BitsAndBytesConfig

from llama_recipes.inference.safety_utils import get_safety_checker, AgentType
from llama_recipes.inference.model_utils import load_model, load_peft_model

from accelerate.utils import is_xpu_available


def load_model(model_name, quantization, use_fast_kernels):
    print(f"use_fast_kernels {use_fast_kernels}")
    if quantization:
        quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    else:
        quantization_config = None

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        return_dict=True,
        quantization_config=quantization_config,
        device_map="auto",
        #local_files_only=True,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa" if use_fast_kernels else None,
    )
    return model

def get_llama_inf_fn(
        model_name,
        peft_model: str = None,
        quantization: bool = False,
        max_new_tokens=100,  # The maximum numbers of tokens to generate
        prompt_file: str = None,
        seed: int = 42,  # seed value for reproducibility
        do_sample: bool = True,  # Whether or not to use sampling ; use greedy decoding otherwise.
        min_length: int = None,  # The minimum length of the sequence to be generated, input prompt + min_new_tokens
        use_cache: bool = True,
        # [optional] Whether or not the model should use the past last key/values attentions Whether or not the model should use the past last key/values attentions (if applicable to the model) to speed up decoding.
        top_p: float = 1.0,
        # [optional] If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to top_p or higher are kept for generation.
        temperature: float = 1.0,  # [optional] The value used to modulate the next token probabilities.
        top_k: int = 50,  # [optional] The number of highest probability vocabulary tokens to keep for top-k-filtering.
        repetition_penalty: float = 1.0,  # The parameter for repetition penalty. 1.0 means no penalty.
        length_penalty: int = 1,
        # [optional] Exponential penalty to the length that is used with beam-based generation.
        max_padding_length: int = None,  # the max padding length to be used with tokenizer padding the prompts.
        use_fast_kernels: bool = False,
        # Enable using SDPA from PyTroch Accelerated Transformers, make use Flash Attention and Xformer memory-efficient kernels
        **kwargs
):
    if is_xpu_available():
        torch.xpu.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)

    torch.manual_seed(seed)

    start_time = time.time()
    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    end_time = time.time()
    print("model loaded in {}".format(end_time - start_time))
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def inference(user_prompt, **kwargs, ):
        # Set the seeds for reproducibility

        batch = tokenizer(user_prompt, return_tensors="pt")
        if is_xpu_available():
            batch = {k: v.to("xpu") for k, v in batch.items()}
        else:
            batch = {k: v.to("cuda") for k, v in batch.items()}

        start = time.perf_counter()
        with torch.no_grad():
            outputs = model.generate(
                **batch,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                top_p=top_p,
                temperature=temperature,
                min_length=min_length,
                use_cache=use_cache,
                top_k=top_k,
                repetition_penalty=repetition_penalty,
                length_penalty=length_penalty,
                pad_token_id=tokenizer.eos_token_id,
                **kwargs
            )
        e2e_inference_time = (time.perf_counter() - start)
        print(f"the inference time is {e2e_inference_time} s")
        output_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output_text
    return inference


def get_llama_sequence_prob_fn(
        model_name,
        peft_model: str = None,
        quantization: bool = False,
        use_fast_kernels: bool = False,
        **kwargs
):
    if torch.xpu.is_available():
        torch.xpu.manual_seed(42)
    else:
        torch.cuda.manual_seed(42)

    torch.manual_seed(42)

    model = load_model(model_name, quantization, use_fast_kernels)
    if peft_model:
        model = load_peft_model(model, peft_model)

    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token

    def get_sequence_prob(prompt, target_sequence):
        # Tokenize the prompt and the target sequence
        prompt_ids = tokenizer.encode(prompt, return_tensors="pt")
        target_ids = tokenizer.encode(target_sequence, return_tensors="pt", add_special_tokens=False)

        device = "xpu" if torch.xpu.is_available() else "cuda"
        prompt_ids = prompt_ids.to(device)
        target_ids = target_ids.to(device)

        # Concatenate prompt and target ids
        input_ids = torch.cat([prompt_ids, target_ids], dim=1)

        # Calculate the probability
        with torch.no_grad():
            outputs = model(input_ids)
            logits = outputs.logits

        # Calculate log probabilities
        log_probs = torch.log_softmax(logits, dim=-1)

        # Calculate the probability of the target sequence
        sequence_log_prob = 0
        for i in range(prompt_ids.shape[1] - 1, input_ids.shape[1] - 1):
            next_token_prob = log_probs[0, i, input_ids[0, i+1]]
            sequence_log_prob += next_token_prob

        # Convert log probability to probability
        sequence_prob = torch.exp(sequence_log_prob).item()

        return sequence_prob

    return get_sequence_prob


def main(model_name="meta-llama/Meta-Llama-3-8B"):
    inference = get_llama_sequence_prob_fn(model_name)
    default_src = "Which company manufactured Colt King Cobra?"
    default_targ1 = "Colt's Manufacturing"
    default_targ2 = "Colt Motorcycles"
    print("Enter your prompt and targets (or press Enter to use defaults):")

    while True:
        src = input("Source: ").strip() or default_src
        targ1 = input("Target 1: ").strip() or default_targ1
        targ2 = input("Target 2: ").strip() or default_targ2

        p1 = inference(src, targ1)
        p2 = inference(src, targ2)

        print(f"{targ1}: {p1}")
        print(f"{targ2}: {p2}")
        print(f"Is second more probable? {p1 < p2}")
        print("==================================\n")


if __name__ == "__main__":
    fire.Fire(main)