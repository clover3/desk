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


def build_toxicity_prompt2(conversation):
    categories = [
        SafetyCategory(
            "Toxicity. ",
            "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."),
    ]
    return build_custom_prompt(
        AgentType.USER,
        create_conversation(conversation),
        categories,
        LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX,
        PROMPT_TEMPLATE_2,
        with_policy=True)


def build_lg2_toxicity_prompt(conversation):
    return build_toxicity_prompt(AgentType.USER,
                                 create_conversation(conversation),
                                 LlamaGuardVersion.LLAMA_GUARD_2)


def load_instruct_as_guard(build_prompt=build_lg2_toxicity_prompt, model_id: str = Llama3_8B_Instruct):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(model_id, device_map="auto")

    def check_conversation(conversation: List[str]) -> tuple[str, float]:
        formatted_prompt = build_prompt(conversation)
        chat = tokenizer.apply_chat_template([{"role": "system", "content": formatted_prompt}], tokenize=False)
        input = tokenizer([chat], return_tensors="pt").to("cuda")
        # print(tokenizer.decode(input["input_ids"][0]))
        prompt_len = input["input_ids"].shape[-1]
        output = model.generate(**input, max_new_tokens=20, pad_token_id=0,
                                return_dict_in_generate=True,
                                output_scores=True)
        generated_seq = output.sequences[0][prompt_len:]
        first_token_score = output.scores[0]
        unsafe_score = first_token_score[0][generated_seq[0]]
        result = tokenizer.decode(output.sequences[0][prompt_len:], skip_special_tokens=True)
        return result, unsafe_score.cpu().numpy()

    return check_conversation

