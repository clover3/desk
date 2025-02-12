from typing import Callable, Dict


class _LLMClientWrapper:
    _instances: Dict = {}

    @classmethod
    def get_client(cls, engine_name: str, max_prompt_len: int = 5000) -> Callable:
        instance_key = f"{engine_name}_{max_prompt_len}"

        if instance_key in cls._instances:
            return cls._instances[instance_key].ask

        if engine_name == "api":
            from llama_user.llama_helper.lf_client import LLMClient
            client = LLMClient(max_prompt_len=max_prompt_len)
        elif engine_name == "llama":
            from llama_user.llama_helper.lf_local import LlamaClient
            client = LlamaClient(max_prompt_len=max_prompt_len)
        elif engine_name == "vllama":
            from llama_user.llama_helper.vllama import VllmClient
            client = VllmClient(max_prompt_len=max_prompt_len)
        else:
            raise ValueError(f"Unknown engine_name: {engine_name}")

        cls._instances[instance_key] = client
        return client.ask


def get_llm_engine_predict_fn(engine_name: str, max_prompt_len: int = 5000) -> Callable:
    return _LLMClientWrapper.get_client(engine_name, max_prompt_len)
