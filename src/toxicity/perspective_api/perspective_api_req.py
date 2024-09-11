import json
import os
import time
from typing import Any
from typing import List, Dict

from datasets import load_dataset
from googleapiclient import discovery
from googleapiclient.errors import HttpError
from tqdm import tqdm

from newbie.path_helper import get_open_ai_mod_perspective_api_res_path


class PerspectiveAPIClient:
    def __init__(
            self, api_key: str,
            max_retries: int = 5, initial_delay: float = 30, backoff_factor: float = 2.0):
        self.client = discovery.build(
            "commentanalyzer",
            "v1alpha1",
            developerKey=api_key,
            discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
            static_discovery=False,
        )

        self.attrib_list: List[str] = [
            "TOXICITY",
            "SEVERE_TOXICITY",
            "IDENTITY_ATTACK",
            "INSULT",
            "PROFANITY",
            "THREAT"
        ]
        self.requestedAttributes: Dict[str, Dict] = {k: {} for k in self.attrib_list}
        self.max_retries = max_retries
        self.initial_delay = initial_delay
        self.backoff_factor = backoff_factor

    def api_call(self, text: str) -> Dict[str, Any]:
        analyze_request = {
            'comment': {'text': text},
            'languages': 'en',
            'requestedAttributes': self.requestedAttributes
        }
        response = self.analyze_with_retry(analyze_request)
        return response

    def analyze_with_retry(self, analyze_request: Dict[str, Any]) -> Dict[str, Any]:
        retries = 0
        delay = self.initial_delay

        while retries < self.max_retries:
            try:
                response = self.client.comments().analyze(body=analyze_request).execute()
                return response
            except HttpError as e:
                if e.resp.status == 429:  # 429 is the status code for quota limit exceeded
                    retries += 1
                    if retries == self.max_retries:
                        raise  # Re-raise the exception if we've exhausted our retries
                    print(f"Quota limit reached. Retrying in {delay} seconds...")
                    time.sleep(delay)
                    delay *= self.backoff_factor  # Increase delay for next potential retry
                else:
                    raise  # Re-raise the exception if it's not a quota limit error

        # This line will only be reached if we've exhausted retries without success
        raise Exception("Max retries reached. Unable to complete request.")


def load_api():
    key_path = os.path.join("newbie", "perspective_api_key.txt")
    return open(key_path, "r").read().strip()


def test_run():
    API_KEY = load_api()
    client = discovery.build(
        "commentanalyzer",
        "v1alpha1",
        developerKey=API_KEY,
        discoveryServiceUrl="https://commentanalyzer.googleapis.com/$discovery/rest?version=v1alpha1",
        static_discovery=False,
    )

    attrib_list = [
        "TOXICITY",
        "SEVERE_TOXICITY",
        "IDENTITY_ATTACK",
        "INSULT",
        "PROFANITY",
        "THREAT"
    ]
    requestedAttributes = {k: {} for k in attrib_list}
    analyze_request = {
        'comment': {'text': 'friendly greetings from python'},
        'requestedAttributes': requestedAttributes
    }
    response = client.comments().analyze(body=analyze_request).execute()
    print(json.dumps(response, indent=2))


def count_lines(file_path: str) -> int:
    with open(file_path, 'r', encoding='utf-8') as f:
        return sum(1 for _ in f)


def process_dataset_and_write_jsonl(output_file: str):
    # Load the dataset
    dataset = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
    client = PerspectiveAPIClient(load_api())

    # Check if the output file exists and count lines
    try:
        processed_items = count_lines(output_file)
        print(f"Resuming from item {processed_items}")
        mode = 'a'  # Append mode
    except FileNotFoundError:
        processed_items = 0
        print("Starting new processing")
        mode = 'w'  # Write mode

    # Open the output file in append or write mode
    with open(output_file, mode, encoding='utf-8') as f:
        # Iterate through the dataset with a progress bar, skipping processed items
        for idx, item in tqdm(enumerate(dataset.skip(processed_items)),
                              initial=processed_items, total=len(dataset),
                              desc="Processing items"):
            prompt = item['prompt']

            # Make the API call
            result = client.api_call(prompt)
            time.sleep(0.5)  # Simulate API latency

            output_item = {
                "prompt": prompt,
                "id": idx,
                "api_result": result
            }
            json.dump(output_item, f, ensure_ascii=False)
            f.write('\n')
            f.flush()


# Helper function to load API key (implement this according to your setup)


# Example usage
if __name__ == "__main__":
    process_dataset_and_write_jsonl(get_open_ai_mod_perspective_api_res_path())
