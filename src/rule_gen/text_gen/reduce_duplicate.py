import json

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path


def main():
    save_path: str = get_rule_gen_save_path("oam", "p2")
    rules = json.load(open(save_path, "r"))
    payload = "\n".join(rules)
    instruction = "Extract all reasons without duplicates. Remove semantic duplicates. Output as list of string json"
    prompt = str(payload) + "\n===\n" + instruction
    response = LLMClient().ask(prompt)
    print(response)


if __name__ == "__main__":
    main()
