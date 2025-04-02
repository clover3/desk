from llama_user.llama_helper.lf_local import LlamaClient2
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token_id = tokenizer.eos_token_id
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")


def get_json_request(prompt):
    return {
        "model": model_name,
        "messages": [
            {"role": "user", "content": prompt}
        ]
    }

def get_parsing_key(tokenizer):
    prompt = "hi"
    t1 = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    t2 = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=False)
    return t1[len(t2):]


prompt = "Say Yes"
response_header = get_parsing_key(tokenizer)
j = get_json_request(prompt)
input_text = tokenizer.apply_chat_template(
            j["messages"], tokenize=False,
            add_generation_prompt=True)
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
output = model.generate(**inputs, max_new_tokens=10,
                              return_dict_in_generate=True,
                              output_scores=True,
                              pad_token_id=pad_token_id)

