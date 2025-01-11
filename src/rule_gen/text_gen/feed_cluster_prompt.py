import os, json
import sys

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path
from rule_gen.open_ai_mod.prompts import why_unsafe


def main():
    cluster_path = os.path.join(output_root_path, "clusters", "KMedoids2.json")
    clusters = json.load(open(cluster_path, "r"))
    rules = []
    for cluster_idx, texts in enumerate(clusters):
        prompt = why_unsafe.format(texts[0])
        response = LLMClient().ask(prompt)
        rules.append(response)

    save_path: str = get_rule_gen_save_path("oam", "KMedois_why")
    json.dump(rules, open(save_path, "w"))


if __name__ == "__main__":
    main()
