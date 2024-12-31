from datetime import datetime
from typing import Dict, Optional
import json

from toxicity.apis.open_ai import get_open_ai
from toxicity.cpath import output_root_path


def format_payload(custom_id, prompt, model, max_tokens):
    return {"custom_id": custom_id,
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {"model": model,
                     "messages": [{
                         "role": "system",
                         "content": "You are a helpful assistant."},
                         {"role": "user",
                          "content": prompt}
                     ],
                     "max_tokens": max_tokens}
            }


import os


def get_batch_payload_path(batch_name):
    base_dir = os.path.join(output_root_path, "batch_request")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{batch_name}_payload.jsonl")


def get_batch_response_path(batch_name):
    base_dir = os.path.join(output_root_path, "batch_request")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{batch_name}_response.jsonl")


def get_batch_response_id_to_text_path(batch_name):
    base_dir = os.path.join(output_root_path, "batch_request")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{batch_name}_res_i2t.jsonl")


def get_path_request_path(batch_name):
    base_dir = os.path.join(output_root_path, "batch_request")
    os.makedirs(base_dir, exist_ok=True)
    return os.path.join(base_dir, f"{batch_name}_requests.json")


def get_gpt_model_name_from_run_name(run_name: str):
    gpt_prefix_list = ["gpt-4o_", "gpt-4o-mini_"]
    for prefix in gpt_prefix_list:
        if run_name.startswith(prefix):
            return prefix[:-1]

    if run_name.startswith("chatgpt_"):
        return "gpt-4o"
    else:
        raise ValueError(run_name + " is not expected")


class BatchChatGPTSender:
    def __init__(self, batch_name: str, model="gpt-4o-mini", max_tokens=2000):
        super().__init__()
        self.batch_name = batch_name
        self.request_file = get_path_request_path(batch_name)
        self.payload_path = get_batch_payload_path(batch_name)
        self.model = model
        print("BatchChatGPTSender - model: " + self.model + " - batch name: " + self.batch_name)
        self._request_queue = []
        self.client = get_open_ai()
        self.client_gb = get_open_ai("GlobalBatch")
        self.max_tokens = max_tokens

    def add_request(self, request_id, prompt: str) -> bool:
        self._request_queue.append((request_id, prompt))
        return True

    def submit_request(self) -> bool:
        if not self._request_queue:
            print("  - No pending requests, returning")
            return True

        try:
            payload = []
            for req_id, prompt in self._request_queue:
                e = format_payload(
                    req_id, prompt, self.model, self.max_tokens
                )
                payload.append(e)

            with open(self.payload_path, "w") as f:
                for e in payload:
                    f.write(json.dumps(e) + "\n")

            print("  - Submitting to OpenAI")
            batch_input_file = self.client.files.create(
                file=open(self.payload_path, "rb"),
                purpose="batch"
            )

            batch_input_file_id = batch_input_file.id
            print("batch_input_file_id", batch_input_file_id)
            self.client.files.wait_for_processing(batch_input_file_id)
            ret = self.client_gb.batches.create(
                input_file_id=batch_input_file_id,
                endpoint="/v1/chat/completions",
                completion_window="24h",
                metadata={
                    "description": self.batch_name
                }
            )
            print(ret)
            print("  - Saving batch info")
            batch_info = {
                "prompts": self._request_queue,
                "timestamp": datetime.now().isoformat(),
                "batch_id": ret.id,
                "batch_input_file_id": batch_input_file_id
            }
            with open(self.request_file, 'w') as f:
                json.dump(batch_info, f)

            self._request_queue = []
            return True

        except Exception as e:
            print(f"Error submitting batch: {e}")
            return False


class BatchChatGPTLoader:
    def __init__(self, batch_name: str):
        self.request_file = get_path_request_path(batch_name)
        self.response_file = get_batch_response_path(batch_name)
        self.response_id2text = get_batch_response_id_to_text_path(batch_name)
        self.prompt_to_response: Optional[Dict[str, str]] = None
        self.custom_id_to_response: Optional[Dict[str, str]] = None
        self.client = get_open_ai()

    def prepare_response(self):
        # if os.path.exists(self.response_file):
        #     print("Read from ", self.response_file)
        #     self.prompt_to_response = json.load(open(self.response_file, 'r'))
        #     self.custom_id_to_response = json.load(open(self.response_id2text, "r"))
        #     return

        batch_info = json.load(open(self.request_file, "r"))
        print("Use batch info ", self.request_file)
        custom_id_to_prompts = dict(batch_info["prompts"])
        batch_id = batch_info["batch_id"]
        batch_status = self.client.batches.retrieve(batch_id)
        if batch_status.status != "completed":
            print("batch_status: ", batch_status.status)
            print(batch_status)
            raise ValueError("Not completed ({})".format(batch_status.status))
        print("Reading {} from OpenAI".format(batch_status.output_file_id))
        content = self.client.files.content(batch_status.output_file_id).text
        prompt_to_response = {}
        custom_id_to_response = {}
        for line in content.split("\n"):
            if line.strip():
                j = json.loads(line)
                response = j['response']['body']['choices'][0]['message']['content']
                if j["response"]["body"]["choices"][0]["message"]["refusal"] is not None:
                    print("Warning the response was refused.")

                custom_id = j['custom_id']
                prompt = custom_id_to_prompts[custom_id]
                custom_id_to_response[custom_id] = response
                prompt_to_response[prompt] = response
        self.custom_id_to_response = custom_id_to_response
        json.dump(custom_id_to_response, open(self.response_id2text, 'w'))

        self.prompt_to_response = prompt_to_response
        json.dump(prompt_to_response, open(self.response_file, 'w'))

    def get_response(self, text: str) -> Optional[str]:
        return self.prompt_to_response[text]
