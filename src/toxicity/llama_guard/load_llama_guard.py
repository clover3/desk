import fire
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion, build_custom_prompt, AgentType, SafetyCategory
from typing import List, Tuple
from enum import Enum
import torch
from llama_recipes.inference.prompt_format_utils import ConversationTurn, LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX,PROMPT_TEMPLATE_2, LLAMA_GUARD_1_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_1
from typing import List, Iterable, Callable, Dict, Tuple, Set, Iterator

from toxicity.llama_guard.llama_guard_formatter import build_toxicity_prompt


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"


def load_llama_guard_model(model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
                           llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2,
                           use_toxicity=False):
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    # model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def check_conversation(conversation: List[str], agent_type: AgentType = AgentType.USER) -> str:
        if not use_toxicity:
            formatted_prompt = build_default_prompt(
                agent_type,
                create_conversation(conversation),
                llama_guard_version)
        else:
            formatted_prompt = build_toxicity_prompt(agent_type,
                create_conversation(conversation),
                llama_guard_version)
        input = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]]
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        result += "\n{0:.4f}".format(unsafe_score.cpu().numpy())
        return result

    return check_conversation


g_model_d = {}
def load_llg2(model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
                           llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2,
                           use_toxicity=True) -> Callable[[list[str]], tuple[str, float]]:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        quantization_config = None

    global g_model_d
    if model_id in g_model_d:
        model = g_model_d[model_id]
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
        g_model_d[model_id] = model

    def check_conversation(conversation: List[str], agent_type: AgentType = AgentType.USER) -> tuple[str, float]:
        if isinstance(conversation, str):
            print("Warning: expected list of string but got string")
        if not use_toxicity:
            formatted_prompt = build_default_prompt(
                agent_type,
                create_conversation(conversation),
                llama_guard_version)
        else:
            formatted_prompt = build_toxicity_prompt(agent_type,
                create_conversation(conversation),
                llama_guard_version)
        input = tokenizer([formatted_prompt], return_tensors="pt").to(device)
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]].cpu().numpy()
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, unsafe_score

    return check_conversation



def load_llg2_custom(category_desc,
                     model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
                     ) -> Callable[[list[str]], tuple[str, float]]:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")

    def check_conversation(conversation: List[str]) -> tuple[str, float]:
        categories = [
            SafetyCategory("Toxicity. ", category_desc),
        ]
        category_short_name_prefix = LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_2
        formatted_prompt = build_custom_prompt(
            AgentType.USER,
            create_conversation(conversation),
            categories,
            category_short_name_prefix,
            prompt_template, with_policy=True)

        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]].cpu().numpy()
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, unsafe_score
    return check_conversation


model_d = {}


def load_llg2_custom2(category_list: list[tuple[str, str]],
                     model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
                     ) -> Callable[[list[str]], tuple[str, float]]:
    quantization_config = BitsAndBytesConfig(load_in_8bit=True)
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    global model_d
    if model_id in model_d:
        model = model_d[model_id]
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quantization_config, device_map="auto")
        model_d[model_id] = model

    def check_conversation(conversation: List[str]) -> tuple[str, float]:
        categories = [
            SafetyCategory(name, desc) for name, desc in category_list
        ]
        category_short_name_prefix = LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_2
        formatted_prompt = build_custom_prompt(
            AgentType.USER,
            create_conversation(conversation),
            categories,
            category_short_name_prefix,
            prompt_template, with_policy=True)

        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=100, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]].cpu().numpy()
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, unsafe_score
    return check_conversation

def main(model_id: str = "meta-llama/Meta-Llama-Guard-2-8B",
         llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2):
    """
    Entry point for Llama Guard inference sample script with interactive mode.
    """
    print("Loading Llama Guard model... This may take a moment.")
    check_conversation = load_llama_guard_model(model_id, llama_guard_version, True)

    while True:
        conversation = []
        print("\nEnter the conversation messages. Press Enter without typing anything to finish the conversation.")

        while True:
            message = input("Enter a message (or press Enter to finish): ").strip()
            if not message:
                break
            conversation.append(message)

        if not conversation:
            print("No conversation entered. Exiting.")
            break

        response = check_conversation(conversation)
        print(f"\nConversation: {conversation}")
        print(f"response: {response}")
        break




if __name__ == "__main__":
    try:
        fire.Fire(main)
    except Exception as e:
        print(e)
