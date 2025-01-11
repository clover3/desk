import os, json
import sys

from desk_util.open_ai import OpenAIChatClient
from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path


def main():
    cluster_path = os.path.join(output_root_path, "clusters", "KMedoids2.json")
    clusters = json.load(open(cluster_path, "r"))
    client = OpenAIChatClient()
    instruction = ("Identify what is topically common in these list of <text>."
                   " Answer in a short phrase. ")
    rules = []
    for cluster_idx, texts in enumerate(clusters):
        payload = "\n".join([f"<text> {t} </text>" for t in texts])
        prompt = str(payload) + "\n===\n" + instruction
        # print(prompt)
        response = client.request(prompt)
        rules.append(response)

    save_path: str = get_rule_gen_save_path("oam", "KMedois30_gpt")
    json.dump(rules, open(save_path, "w"))


# Conclusion: Just using KMedoids makes too many non-relevant being included.
# KMeans alone is

if __name__ == "__main__":
    main()
