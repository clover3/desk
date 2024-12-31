from typing import List

from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion, \
    AgentType
from transformers import AutoTokenizer

from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.llama_guard.llama_guard_formatter import build_toxicity_prompt
from llama_user.llama_helper import Llama3_8B_Instruct


def llama3_instruct(llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2,
                    use_toxicity=True) -> None:
    tokenizer = AutoTokenizer.from_pretrained(Llama3_8B_Instruct)

    def check_conversation(conversation: List[str], agent_type: AgentType = AgentType.USER):
        if not use_toxicity:
            formatted_prompt = build_default_prompt(
                agent_type,
                create_conversation(conversation),
                llama_guard_version)
        else:
            formatted_prompt = build_toxicity_prompt(agent_type,
                                                     create_conversation(conversation),
                                                     llama_guard_version)

        print(formatted_prompt)

    def inst_formatting(text):
        content = build_toxicity_prompt(AgentType.USER,
                                        create_conversation([text]),
                                        llama_guard_version)
        input_ids = tokenizer.apply_chat_template([{"role": "user", "content": content}])
        input = tokenizer.apply_chat_template([{"role": "user", "content": content}], return_tensors="pt").to("cuda")
        print(tokenizer.decode(input_ids))
        print(input)

    split = "train"
    test_dataset: ToxigenBinary = ToxigenBinary(split)

    for e in test_dataset:
        print(e)
        inst_formatting(e['text'])
        break


if __name__ == "__main__":
    llama3_instruct()
