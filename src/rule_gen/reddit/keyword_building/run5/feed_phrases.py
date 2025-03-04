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
    save_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram", f"{sb}.json")
    d = json.load(open(save_path, "r"))
    pos = d["pos"]
    ngrams = set()
    for ngram_per in pos:
        ngram_per = [t for t in ngram_per if t]
        for phrase, prob in ngram_per:
            ngrams.add(phrase)
        # ngrams.add([phrase for phrase, prob in ngram_per])
    prompt = get_instruction(ngrams)
    ret_text = client.request(prompt)
    print(ret_text)
    output = [ret_text]
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "ngram_based_j_raw", f"{sb}.json")
    make_parent_exists(res_save_path)
    json.dump(output, open(res_save_path, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
