from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

class LlamaClient:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
        self.max_prompt_len = max_prompt_len

    def len_filter(self, text):
        if text is None:
            return text
        if len(text) < self.max_prompt_len:
            return text
        else:
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
        response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return response.split("Assistant:")[-1].strip()

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
