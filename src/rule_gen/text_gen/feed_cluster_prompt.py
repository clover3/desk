import os, json
import sys

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path
from rule_gen.open_ai_mod.prompts import why_unsafe



def generate_save_rule(prompt_fmt, request_fn, run_name):
    cluster_path = os.path.join(output_root_path, "clusters", "KMedoids2.json")
    clusters = json.load(open(cluster_path, "r"))
    rules = []
    for cluster_idx, texts in enumerate(clusters):
        t = texts[0]
        prompt = prompt_fmt.format(t)
        response = request_fn(prompt)
        rules.append(response)
    save_path: str = get_rule_gen_save_path("oam", run_name)
    json.dump(rules, open(save_path, "w"))


def short_keys():
    ask = LLMClient().ask
    run_name = "KMedois_why"
    generate_save_rule(why_unsafe, ask, run_name)


def verbose_one():
    prompt_fmt = """
    Below is a text that is categorized unsafe as
     inputs or outputs of LLMs.
    Guess why the text is consider the text unsafe.
    Answer in a format `Content that ~'.
    If for some reason you think this text should 
    not have been classified unsafe.
    Answer with ‘N/A’.
    Here’s the text: {}
    Your list:
    """
    ask = LLMClient().ask
    run_name = "KMedois_content"
    generate_save_rule(prompt_fmt, ask, run_name)


def main():
    verbose_one()


if __name__ == "__main__":
    main()
