import os
import sys
import requests


class LLMClient:
    def __init__(self, host=None, port=8000,
                 max_prompt_len=10000,
                 model_name="dummy",
                 ):
        if host is None:
            try:
                host = os.environ["API_HOST"]
            except KeyError:
                host = "localhost"
                print(f"Environment variable API_HOST is not found default to {host}")
        self.url = f"http://{host}:{port}/v1/chat/completions"
        self.max_prompt_len = max_prompt_len
        self.model_name = model_name

    def len_filter(self, text):
        if text is None:
            return text

        if len(text) < self.max_prompt_len:
            return text
        else:
            print("Text has {} characters. Truncate to {}".format(len(text), self.max_prompt_len))
            return text[:self.max_prompt_len]

    def _ask_no_system(self, prompt):
        ret = requests.post(self.url, json=j)
        content = ret.json()["choices"][0]["message"]["content"]
        return content

    def ask(self, prompt, system_prompt=None):
        prompt = self.len_filter(prompt)
        system_prompt = self.len_filter(system_prompt)
        j = self.get_json_request(prompt, system_prompt)
        ret = requests.post(self.url, json=j)
        content = ret.json()["choices"][0]["message"]["content"]
        return content

    def get_json_request(self, prompt, system_prompt):
        if system_prompt is None:
            j = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        else:
            j = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "system",
                        "content": system_prompt
                    },
                    {
                        "role": "user",
                        "content": prompt
                    }
                ]
            }
        return j


def transform_text_by_llm(instruction, text_list: list[str]) -> list[str]:
    client = LLMClient()

    def convert(text):
        prompt = f"{instruction}\n{text}"
        return client.ask(prompt)

    # print(response)
    for t in text_list:
        print("Input:", t)
        r = convert(t)
        print("Output:", r)
        yield r


def main():
    response = LLMClient().ask(sys.argv[1])
    print(response)


if __name__ == "__main__":
    main()

