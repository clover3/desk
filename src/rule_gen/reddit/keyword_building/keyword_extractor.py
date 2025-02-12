import json
from typing import List

from desk_util.open_ai import OpenAIChatClient


def parse_openai_json(s):
    lines = s.split("\n")
    if lines[0] == "```json" and lines[-1] == "```":
        raw_j = "\n".join(lines[1:-1])
        return json.loads(raw_j)
    if lines[0] == "```" and lines[-1] == "```":
        raw_j = "\n".join(lines[1:-1])
        return json.loads(raw_j)
    if lines[2] == "```json" and lines[-1] == "```":
        raw_j = "\n".join(lines[3:-1])
        return json.loads(raw_j)
    elif lines[0] == "[" and lines[-1] == "]":
        raw_j = s
        return json.loads(raw_j)
    elif lines[0] == "{" and lines[-1] == "}":
        raw_j = s
        j = json.loads(raw_j)
        key = next(iter(j.keys()))
        return j[key]
    else:
        print("Parse failed", lines)
        raise ValueError()


class KeywordExtractor:
    def __init__(self, client=None):
        if client is None:
            client = OpenAIChatClient("gpt-4o")

        self.client = client
        self.instruction = "Extract keywords from the following text. Return as a json list"

    def extract_keywords(self, rule_text: str) -> List[str]:
        prompt = f"{self.instruction}\n<text>{rule_text}</text>"
        ret_text = self.client.request(prompt)
        ret = parse_openai_json(ret_text)
        if not ret:
            print("warning nothing parsed", ret_text)
            print("ret_text: ", ret_text)
            print("Corresponding rule: ", rule_text)
        return ret
