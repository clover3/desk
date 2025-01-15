from rule_gen.open_ai_mod.train_proto import get_data_arguments
from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg
import os, json

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path


def enum_pos_items():
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(False)
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    texts = [item for item in train_dataset if item['label']]
    for t in texts[:30]:
        yield t["text"]


def generate_save_rule(prompt_fmt, request_fn, run_name):
    rules = []
    for t in enum_pos_items():
        prompt = prompt_fmt.format(t)
        response = request_fn(prompt)
        rules.append(response)
    save_path: str = get_rule_gen_save_path("oam", run_name)
    json.dump(rules, open(save_path, "w"))


def make_short_phrase():
    prompt_fmt = """
    Below is a text that is categorized unsafe as
     inputs or outputs of LLMs.
    Guess why the text is consider the text unsafe.
    Answer in a short phrase.
    If for some reason you think this text should 
    not have been classified unsafe.
    Answer with ‘N/A’.
    Here’s the text: {}
    Your list:
    """
    run_name = "p1"
    client_fn = LLMClient()
    request_fn = client_fn.ask
    generate_save_rule(prompt_fmt, request_fn, run_name)


def make_definition():
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
    run_name = "p2"
    client_fn = LLMClient()
    request_fn = client_fn.ask
    generate_save_rule(prompt_fmt, request_fn, run_name)


if __name__ == "__main__":
    make_definition()
