from llama_recipes.inference.prompt_format_utils import LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2

from typing import List

from llama_recipes.inference.prompt_format_utils import create_conversation, LlamaGuardVersion, \
    build_custom_prompt, SafetyCategory
from transformers import AutoTokenizer, AutoModelForCausalLM

from toxicity.llama_guard.llama_guard_formatter import build_toxicity_prompt, AgentType
from toxicity.llama_helper.llama_model_names import Llama3_8B_Instruct
from typing import List

from llama_recipes.inference.prompt_format_utils import LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_2
from llama_recipes.inference.prompt_format_utils import create_conversation, LlamaGuardVersion, \
    build_custom_prompt, SafetyCategory
from transformers import AutoTokenizer, AutoModelForCausalLM

from toxicity.llama_guard.llama_guard_formatter import build_toxicity_prompt, AgentType
from toxicity.llama_helper.llama_model_names import Llama3_8B_Instruct


def load_ilu(run_name):
    model_id = "Orenguteng/Llama-3.1-8B-Lexi-Uncensored-V2"
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")
    unsafe_idx = tokenizer.vocab["unsafe"]
    def check_conversation_ilu_1_2(conversation: List[str]) -> tuple[str, float]:
        if run_name == "ilu1":
            formatted_prompt = build_toxicity_prompt(AgentType.USER,
                                                     create_conversation(conversation),
                                                     LlamaGuardVersion.LLAMA_GUARD_2)
        elif run_name == "ilu2":
            formatted_prompt = build_toxicity_prompt(AgentType.USER,
                                                     create_conversation(conversation),
                                                     LlamaGuardVersion.LLAMA_GUARD_3)
        else:
            raise ValueError()
        chat = tokenizer.apply_chat_template([{"role": "user", "content": formatted_prompt}], tokenize=False)
        input = tokenizer([chat], return_tensors="pt").to("cuda")
        print(tokenizer.decode(input["input_ids"][0]))
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=20, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][unsafe_idx]
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, unsafe_score.cpu().numpy()

    def check_conversation_ilu3(conversation: List[str]) -> tuple[str, float]:
        if run_name == "ilu3":
            formatted_prompt = build_toxicity_prompt(AgentType.USER,
                                                     create_conversation(conversation),
                                                     LlamaGuardVersion.LLAMA_GUARD_3)
        else:
            raise ValueError()
        input = tokenizer([formatted_prompt], return_tensors="pt").to("cuda")
        print("=== input ===")
        print(tokenizer.decode(input["input_ids"][0]))
        print("====")
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=20, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]]
        out_ids = output.sequences[0][prompt_len:]
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        print("out_ids", out_ids)
        print("=== result ===")
        print(result)
        print("======")
        return result, unsafe_score.cpu().numpy()

    if run_name == "ilu1" or run_name == "ilu2":
        return check_conversation_ilu_1_2
    elif run_name == "ilu3":
        return check_conversation_ilu3
    else:
        raise ValueError()
