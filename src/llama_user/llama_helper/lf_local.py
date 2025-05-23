import numpy as np
import math

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


def get_parsing_key(tokenizer):
    prompt = "hi"
    t1 = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                       add_generation_prompt=True)
    t2 = tokenizer.apply_chat_template([{"role": "user", "content": prompt}], tokenize=False,
                                        add_generation_prompt=False)
    return t1[len(t2):]


global_model_d = {}

class LlamaClient:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000):
        global global_model_d
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_name not in global_model_d:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            global_model_d[model_name] = self.model
        else:
            self.model = global_model_d[model_name]
        self.max_prompt_len = max_prompt_len
        self.response_header = get_parsing_key(self.tokenizer)
        if not self.response_header:
            raise KeyError()
        self.truncate_count = 0

    def len_filter(self, text):
        if text is None:
            return text

        if len(text) < self.max_prompt_len:
            return text

        self.truncate_count += 1
        if self.truncate_count == 1 or math.log10(self.truncate_count).is_integer():
            print(f"Text has {len(text)} characters. Truncate to {self.max_prompt_len}")

        return text[:self.max_prompt_len]

    def ask(self, prompt, system_prompt=None):
        prompt = self.len_filter(prompt)
        system_prompt = self.len_filter(system_prompt)
        j = self.get_json_request(prompt, system_prompt)
        return self._generate_response(j)

    def _generate_response(self, request_json):
        input_text = self.tokenizer.apply_chat_template(
            request_json["messages"], tokenize=False,
            add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        outputs = self.model.generate(**inputs, max_new_tokens=10,
                                      pad_token_id=pad_token_id)
        gen_text = self.tokenizer.decode(outputs[0] )
        idx = gen_text.find(self.response_header)
        if idx == -1:
            raise ValueError()
        response = gen_text[idx+len(self.response_header):]

        return response
    def get_json_request(self, prompt, system_prompt):
        if system_prompt is None:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }


class LlamaClient2:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000):
        global global_model_d
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        if model_name not in global_model_d:
            self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
            global_model_d[model_name] = self.model
        else:
            self.model = global_model_d[model_name]
        self.max_prompt_len = max_prompt_len
        self.response_header = get_parsing_key(self.tokenizer)
        if not self.response_header:
            raise KeyError()
        self.truncate_count = 0

    def len_filter(self, text):
        if text is None:
            return text

        if len(text) < self.max_prompt_len:
            return text

        self.truncate_count += 1
        if self.truncate_count == 1 or math.log10(self.truncate_count).is_integer():
            print(f"Text has {len(text)} characters. Truncate to {self.max_prompt_len}")

        return text[:self.max_prompt_len]

    def ask(self, prompt, system_prompt=None) -> tuple[str, float]:
        prompt = self.len_filter(prompt)
        system_prompt = self.len_filter(system_prompt)
        j = self.get_json_request(prompt, system_prompt)
        return self._generate_response(j)

    def _generate_response(self, request_json) -> tuple[str, float]:
        input_text = self.tokenizer.apply_chat_template(
            request_json["messages"], tokenize=False,
            add_generation_prompt=True)
        inputs = self.tokenizer(input_text, return_tensors="pt").to(self.model.device)
        prompt_len = inputs["input_ids"].shape[-1]
        pad_token_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else self.tokenizer.eos_token_id
        output = self.model.generate(**inputs, max_new_tokens=10,
                                      return_dict_in_generate=True,
                                      output_scores=True,
                                      pad_token_id=pad_token_id)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        confidence = first_token_score[0][generated_seq[0]]
        transition_scores = self.model.compute_transition_scores(
            output.sequences, output.scores, normalize_logits=True
        )
        confidence = transition_scores[0][0]

        prob = np.exp(confidence.cpu().numpy())
        gen_text = self.tokenizer.decode(generated_seq)
        return gen_text, float(confidence.cpu().numpy())
        # return gen_text, float(prob)

    def get_json_request(self, prompt, system_prompt):
        if system_prompt is None:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "user", "content": prompt}
                ]
            }
        else:
            return {
                "model": self.model.config.name_or_path,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
                ]
            }

#
# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
#
#
# class LlamaClient:
#     def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name)
#         self.model = LLM(model=model_name)  # vLLM optimized model
#         self.model_name = model_name
#         self.max_prompt_len = max_prompt_len
#
#     def len_filter(self, text):
#         if text is None:
#             return text
#         if len(text) < self.max_prompt_len:
#             return text
#         else:
#             print(f"Text has {len(text)} characters. Truncate to {self.max_prompt_len}")
#             return text[:self.max_prompt_len]
#
#     def ask(self, prompt, system_prompt=None):
#         prompt = self.len_filter(prompt)
#         system_prompt = self.len_filter(system_prompt)
#         j = self.get_json_request(prompt, system_prompt)
#         return self._generate_response(j)
#
#     def _generate_response(self, request_json):
#         messages = request_json["messages"]
#         input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
#
#         sampling_params = SamplingParams(max_tokens=200, temperature=0.7, top_p=0.9)  # Adjust parameters as needed
#         outputs = self.model.generate([input_text], sampling_params)
#
#         response = outputs[0].outputs[0].text  # Extract generated text
#         return response.strip()
#
#     def get_json_request(self, prompt, system_prompt):
#         if system_prompt is None:
#             return {
#                 "model": self.model_name,
#                 "messages": [
#                     {"role": "user", "content": prompt}
#                 ]
#             }
#         else:
#             return {
#                 "model": self.model_name,
#                 "messages": [
#                     {"role": "system", "content": system_prompt},
#                     {"role": "user", "content": prompt}
#                 ]
#             }
