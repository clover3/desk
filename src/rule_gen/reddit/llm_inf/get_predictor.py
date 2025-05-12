from typing import Callable


def get_llm_predictor(run_name) -> Callable[[str], tuple[int, str, float]]:
    if run_name.startswith("llama"):
        from rule_gen.reddit.s9.token_scoring import get_llama_criteria_scorer
        return get_llama_criteria_scorer(run_name)
    elif run_name == "llg_default" or run_name == "llg_toxic":
        use_toxicity = run_name == "llg_toxic"
        from toxicity.llama_guard.load_llama_guard import load_llg2
        llama_guard_fn = load_llg2(use_toxicity=use_toxicity)
        def predict(text):
            text_out, score = llama_guard_fn([text])
            pred = int("unsafe" in text_out)
            return pred, text_out, float(score)
        return predict
    elif run_name.startswith("chatgpt"):
        from rule_gen.reddit.llm_inf.gpt4o_inf_triplet_api import get_gpt4o_predictor
        return get_gpt4o_predictor(run_name)

    else:
        raise ValueError()
