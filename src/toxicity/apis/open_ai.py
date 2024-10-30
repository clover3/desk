import json
import os
from openai import OpenAI
import logging

from openai.lib.azure import AzureOpenAI

from toxicity.cpath import data_root_path
from toxicity.cpath import output_root_path


def setup_file_logger(logger_name):
    log_file = os.path.join(output_root_path, "log", "{}.log".format(logger_name))
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    logger.propagate = False

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                                  datefmt='%m-%d %H:%M:%S',
                                  )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


class OpenAIChatClient:
    def __init__(self, model="gpt-4o-mini"):
        self.logger = setup_file_logger("openai")
        self.client = get_open_ai()
        self.model = model

    def request(self, message, ):
        chat_completion = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": message}]
        )
        msg = {
            "request": message,
            "response": chat_completion.choices[0].message.content
        }
        self.logger.info(json.dumps(msg))
        return chat_completion.choices[0].message.content


# Example usage:
# client = OpenAIChatClient()
# response = client.request("Hello world")
# print(response)
def get_open_ai_my():
    api_key_path = os.path.join(data_root_path, "openai_api_key.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()

    return OpenAI(api_key=key)


def get_open_ai():
    api_key_path = os.path.join(data_root_path, "openai_api_key_uva.txt")
    with open(api_key_path, "r") as f:
        key = f.read().strip()
    azure_endpoint = "https://rtp2-shared.openai.azure.com"
    client = AzureOpenAI(api_key=key, api_version="2024-10-21", azure_endpoint=azure_endpoint)
    return client
