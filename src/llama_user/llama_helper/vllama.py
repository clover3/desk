from vllm import LLM, SamplingParams


class VllmClient:
    def __init__(self, model_name="meta-llama/Meta-Llama-3-8B-Instruct", max_prompt_len=10000):
        self.llm = LLM(model=model_name, dtype="float16")
        self.max_prompt_len = max_prompt_len
        self.sampling_params = SamplingParams(
            max_tokens=10,
            temperature=0.0  # Equivalent to greedy decoding
        )

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
        conversation = self.get_conversation(prompt, system_prompt)
        outputs = self.llm.chat(conversation,
                           self.sampling_params,
                           use_tqdm=False)
        request_output = outputs[0]
        completion_output = request_output.outputs[0]
        generated_text = completion_output.text
        return generated_text

    def get_conversation(self, prompt, system_prompt):
        if system_prompt is None:
            conversation =  [
                    {"role": "user", "content": prompt}
                ]
        else:
            conversation = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}
            ]
        return conversation