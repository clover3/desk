import json

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.reddit.path_helper import get_reddit_rule_path


def get_instruction(sb="askscience"):
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text = " ".join([r['summary'] for r in rules])
    inst_summary = "The above lists prohibited contents, classify if the following text is prohibited. "
    inst_summary += "If prohibited, output 'unsafe' as a first token. If not, output 'safe'"
    return rule_text + "\n " + inst_summary


def main():
    client = LLMClient()
    instruction = get_instruction()
    print(instruction)
    text = "Does that mean that I could take a baby born 200 000 years ago, bring him up in todays world and he would be able to intellectualy compete with kids born today?"
    text = "Andddddd we've opened that can of worms I see..."
    print(text)
    ret = client.ask(text, instruction)
    print(ret)


if __name__ == "__main__":
    main()
