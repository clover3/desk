import os
import random
from typing import Callable

import yaml
from omegaconf import OmegaConf

from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import prompt_conf_root
from rule_gen.reddit.classifier_loader.inst_builder import get_instruction_from_run_name

gpt_prefix_list = ["chatgpt_", "gpt-4o_", "gpt-4o-mini"]


class PromptBuilder:
    def __init__(self, run_name):
        self.run_name = run_name
        self.max_text_len = 2000
        run_name_for_inst = None
        for prefix in gpt_prefix_list:
            if run_name.startswith(prefix):
                run_name_for_inst = run_name.replace(prefix, "api_")
                break

        if run_name_for_inst is None:
            raise ValueError()

        instruction, pos_keyword = get_instruction_from_run_name(run_name_for_inst)
        self.instruction = instruction
        self.pos_keyword = pos_keyword

    def get_prompt(self, text):
        prompt = self.instruction + "\n" + text[:self.max_text_len]
        return prompt

    def get_label_from_response(self, response):
        pred = self.pos_keyword in response.lower()
        ret = int(pred)
        return ret


class NumberAdder:
    def __init__(self, msg="Final number:"):
        self.number = 0
        self.msg = msg

    def add(self, value):
        self.number += value

    def __del__(self):
        print(f"{self.msg} {self.number}")


def dummy_counter(run_name):
    run_name = run_name[len("dummy_"):]
    instruction, pos_keyword = get_instruction_from_run_name(run_name)
    print(instruction)
    adder = NumberAdder()
    def predict(text):
        print(len(instruction), len(text))
        n_char = len(instruction) + len(text)
        n_char = min(n_char, 5000)
        adder.add(n_char)
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


def load_local_based(run_name):
    max_text_len = 5000
    from llama_user.llama_helper.lf_local import LlamaClient

    run_name = run_name.replace("llama_", "api_")
    instruction, pos_keyword = get_instruction_from_run_name(run_name)

    client = LlamaClient(max_prompt_len=5000)

    def predict(text):
        text = text[:max_text_len]
        if isinstance(instruction, str):
            prompt = instruction + "\n" + text[:max_text_len]
        else:
            prompt = instruction(text)
        ret_text = client.ask(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def load_api_based(run_name):
    max_text_len = 5000
    from llama_user.llama_helper.lf_client import LLMClient
    client = LLMClient(max_prompt_len=5000)
    instruction, pos_keyword = get_instruction_from_run_name(run_name)

    def predict(text):
        text = text[:max_text_len]
        ret_text = client.ask(text, instruction)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def load_api_based2(run_name):
    max_text_len = 5000
    from llama_user.llama_helper.lf_client import LLMClient
    client = LLMClient(max_prompt_len=5000)
    run_name = run_name.replace("api2_", "api_")
    instruction, pos_keyword = get_instruction_from_run_name(run_name)

    def predict(text):
        text = text[:max_text_len]
        if isinstance(instruction, str):
            prompt = instruction + "\n" + text[:max_text_len]
        else:
            prompt = instruction(text)
        ret_text = client.ask(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def load_chatgpt_based(run_name) -> Callable[[str], tuple[int, float]]:
    from desk_util.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")
    run_name = run_name.replace("chatgpt_", "api_")
    instruction, pos_keyword = get_instruction_from_run_name(run_name)
    max_prompt_len = 5000

    def predict(text):
        if isinstance(instruction, str):
            prompt = instruction + "\n" + text[:max_prompt_len]
        else:
            prompt = instruction(text)
        ret_text = client.request(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def load_from_conf(run_name):
    prefix = "conf_"
    assert run_name.startswith(prefix)
    conf_name = run_name[len(prefix):]
    conf_path = os.path.join(prompt_conf_root, f"{conf_name}.yaml")
    conf = OmegaConf.load(conf_path)

    if conf["client"] == "gpt":
        client = OpenAIChatClient("gpt-4o")
    elif conf["client"] == "llama":
        from llama_user.llama_helper.lf_client import LLMClient
        client = LLMClient(max_prompt_len=5000)
    else:
        raise ValueError("Conf {} is not expected".format(conf["client"]))

    pos_keyword = conf["pos_keyword"]
    inst_fmt = conf["inst_fmt"]
    debug = conf.get("debug", False)
    max_text_len = 5000

    def predict(text):
        prompt = inst_fmt.format(text[:max_text_len])
        ret_text = client.request(prompt)
        pred = pos_keyword.lower() in ret_text.lower()
        if debug:
            print(prompt)
            print("ret:", ret_text)
        ret = int(pred)
        return ret, 0

    return predict

