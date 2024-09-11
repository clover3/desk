from enum import Enum
from typing import List

from llama_recipes.inference.prompt_format_utils import ConversationTurn, LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX, \
    PROMPT_TEMPLATE_2, LLAMA_GUARD_1_CATEGORY_SHORT_NAME_PREFIX, PROMPT_TEMPLATE_1
from llama_recipes.inference.prompt_format_utils import build_default_prompt, create_conversation, LlamaGuardVersion, \
    build_custom_prompt, SafetyCategory


class AgentType(Enum):
    AGENT = "Agent"
    USER = "User"


def build_toxicity_prompt(
        agent_type: AgentType,
        conversations: List[ConversationTurn],
        llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2):
    categories = [
        SafetyCategory(
            "Toxicity. ",
            "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."),
    ]
    if llama_guard_version == LlamaGuardVersion.LLAMA_GUARD_2:
        category_short_name_prefix = LLAMA_GUARD_2_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_2
    else:
        category_short_name_prefix = LLAMA_GUARD_1_CATEGORY_SHORT_NAME_PREFIX
        prompt_template = PROMPT_TEMPLATE_1

    return build_custom_prompt(
        agent_type,
        conversations,
        categories,
        category_short_name_prefix,
        prompt_template, with_policy=True)


class LlamaGuardFormatter:
    def __init__(self,
                 safe_str: str = "safe",
                 unsafe_str: str = "unsafe\nS1",
                 agent_type: AgentType = AgentType.USER,
                 llama_guard_version: LlamaGuardVersion = LlamaGuardVersion.LLAMA_GUARD_2,
                 use_toxicity=True
                 ):
        self.safe_str = safe_str
        self.unsafe_str = unsafe_str
        self.agent_type = agent_type
        self.llama_guard_version = llama_guard_version
        self.use_toxicity = use_toxicity

    def get_prompt(self, conversation: List[str]) -> str:
        if not self.use_toxicity:
            formatted_prompt = build_default_prompt(
                self.agent_type,
                create_conversation(conversation),
                self.llama_guard_version)
        else:
            formatted_prompt = build_toxicity_prompt(self.agent_type,
                                                     create_conversation(conversation),
                                                     self.llama_guard_version)
        return formatted_prompt

    def get_label_str(self, label: int):
        if label:
            return self.unsafe_str
        else:
            return self.safe_str
