import json

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
import os


def get_instruction(ngram_list):
    instruction = str(ngram_list)
    instruction += "\n"
    instruction += ("Find what are the common characteristics of the above phrases. "
                    "Briefly describe criteria with examples. Answer in list of string json format. ")
    instruction += "\n"
    return instruction


def main(sb="fantasyfootball"):
    from desk_util.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")
    j_save_path = os.path.join(
        output_root_path, "reddit",
        "ngram_based_j_raw", f"{sb}.json")
    d = json.load(open(j_save_path, "r"))
    prompt = "Compress each text in the list by skipping 'phrase', 'many'.\n"
    prompt += str(d)
    ret_text = client.request(prompt)
    save_path = os.path.join(
        output_root_path, "reddit",
        "ngram_based_j2", f"{sb}.json")
    make_parent_exists(save_path)
    with open(save_path, "w") as f:
        f.write(ret_text)


if __name__ == "__main__":
    fire.Fire(main)
