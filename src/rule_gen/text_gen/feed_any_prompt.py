from rule_gen.open_ai_mod.train_proto import get_data_arguments
from rule_gen.reddit.train_common import ClfDatasetLoader, get_datasets_from_dataset_arg
import os, json
import sys

from llama_user.llama_helper.lf_client import LLMClient
from rule_gen.cpath import output_root_path
from rule_gen.open_ai_mod.path_helper import get_rule_gen_save_path
from rule_gen.open_ai_mod.prompts import why_unsafe, why_unsafe_key


def enum_pos_items():
    dataset_builder = ClfDatasetLoader()
    dataset_args = get_data_arguments(False)
    train_dataset, eval_dataset = get_datasets_from_dataset_arg(dataset_builder, dataset_args)
    texts = [item for item in train_dataset if item['label']]
    for t in texts[:30]:
        yield t["text"]


def main():
    rules = []
    for t in enum_pos_items():
        prompt = why_unsafe_key.format(t)
        response = LLMClient().ask(prompt)
        rules.append(response)

    save_path: str = get_rule_gen_save_path("oam", "p1")
    json.dump(rules, open(save_path, "w"))


if __name__ == "__main__":
    main()
