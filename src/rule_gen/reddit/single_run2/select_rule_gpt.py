import json

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
import os
from rule_gen.reddit.path_helper import get_reddit_rule_path


def get_rule_sel_instruction(sb):
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    instruction = str(rules)
    instruction += "\n"
    instruction += "If the following text is deleted, which of the above rule is most appropriate? "
    instruction += f"Only output the number of the rule as the first token. "
    instruction += "\n"
    return instruction


def main(sb="fantasyfootball"):
    from desk_util.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")

    instruction = get_rule_sel_instruction(sb)
    min_sel_diff_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel", "", f"{sb}.csv")
    todo = json.load(open(min_sel_diff_path, "r"))

    output = []
    for j in todo:
        text_j = {"text": j['text']}
        prompt = instruction + json.dumps(text_j)
        ret_text = client.request(prompt)
        text_j["response"] = ret_text
        print(ret_text)
        output.append(text_j)

    res_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel_gpt_res", f"{sb}.json")
    make_parent_exists(res_save_path)
    json.dump(output, open(res_save_path, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
