from typing import Callable

from toxicity.llama_guard.load_llama_guard import load_llg2_custom, load_llg2_custom2, load_llg2


def load_llama_guard_toxicity(name) -> Callable[[str], tuple[int, float]]:
    inst_toxic = "Toxicity is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    context = "This is a reddit comment, where the previous context is missing."
    desc = inst_toxic + " " + context
    def predict(text):
        fn = load_llg2_custom(desc)
        text_out, score = fn([text])
        return "unsafe" in text_out, score
    return predict


def load_categories_for_llg(name):
    raise NotImplementedError()
    inst_toxic = ("Toxicity is a rude, disrespectful, "
                  "or unreasonable comment that is likely to make people leave a discussion.")
    context = "This is a reddit comment, where the previous context is missing."
    desc = inst_toxic + " " + context
    category_list: list[tuple[str, str]] = [("Toxicity", desc)]
    return category_list


def build_toxicity_category():
    inst_toxic = ("Toxicity is a rude, disrespectful, "
                  "or unreasonable comment that is likely to make people leave a discussion.")
    context = "This is a reddit comment, where the previous context is missing."
    desc = inst_toxic + " " + context
    category_list: list[tuple[str, str]] = [("Toxicity", desc)]
    return category_list


def build_toxicity_sb_aware(sb):
    inst_toxic = ("Toxicity is a rude, disrespectful, "
                  "or unreasonable comment that is likely to make people leave a discussion.")
    context = "This is a reddit comment posted in the {} subreddit, where the previous context is missing.".format(sb)
    desc = inst_toxic + " " + context
    category_list: list[tuple[str, str]] = [("Toxicity", desc)]
    return category_list


def load_llama_guard_based(name) -> Callable[[str], tuple[int, float]]:
    is_default_sb = name.startswith("llg_default_")
    if name == "llg_toxic":
        category_list = build_toxicity_category()
    elif name.startswith("llg_toxic_"):
        tokens = name.split("_")
        sb = "_".join(tokens[2:])
        category_list = build_toxicity_sb_aware(sb)
    elif name == "llg_default" or is_default_sb:
        category_list = []
    else:
        category_list = load_categories_for_llg(name)

    if name == "llg_default" or is_default_sb:
        fn = load_llg2(use_toxicity=False)
    else:
        fn = load_llg2_custom2(category_list)

    if is_default_sb:
        tokens = name.split("_")
        sb = "_".join(tokens[2:])
        note = "(This text is written in the {} subreddit of Reddit.)".format(sb)
        def predict(text):
            text = note + text
            text_out, score = fn([text])
            return int("unsafe" in text_out), score
    else:
        def predict(text):
            text_out, score = fn([text])
            return int("unsafe" in text_out), score
    return predict

