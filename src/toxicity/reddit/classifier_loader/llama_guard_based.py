from typing import Callable

from toxicity.llama_guard.load_llama_guard import load_llg2_custom


def load_llama_guard_toxicity(name) -> Callable[[str], tuple[int, float]]:
    inst_toxic = "Toxicity is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    context = "This is a reddit comment, where the previous context is missing."
    desc = inst_toxic + " " + context
    def predict(text):
        fn = load_llg2_custom(desc)
        text_out, score = fn([text])
        return "unsafe" in text_out, score
    return predict


def load_llama_guard_based(name) -> Callable[[str], tuple[int, float]]:
    inst_toxic = "Toxicity is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    context = "This is a reddit comment, where the previous context is missing."
    desc = inst_toxic + " " + context
    def predict(text):
        fn = load_llg2_custom(desc)
        text_out, score = fn([text])
        return "unsafe" in text_out, score
    return predict

