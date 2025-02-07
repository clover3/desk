import random
from typing import Callable

from rule_gen.reddit.classifier_loader.prompt_based import load_from_conf, load_api_based2, load_local_based


def get_random_classifier():
    def predict(text):
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


def get_always_one_clf():
    def predict(text):
        return 1, 0
    return predict


def get_classifier(run_name) -> Callable[[str], tuple[int, float]]:
    if run_name.startswith("bert"):
        from rule_gen.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
        return get_classifier_pipeline(run_name)
    elif run_name == "random":
        return get_random_classifier()
    elif run_name == "always_one":
        return get_always_one_clf()
    elif run_name.startswith("dummy_"):
        from rule_gen.reddit.classifier_loader.prompt_based import dummy_counter
        return dummy_counter(run_name)
    elif run_name.startswith("api_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_api_based
        return load_api_based(run_name)
    elif run_name.startswith("api2_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_api_based2
        return load_api_based2(run_name)
    elif run_name.startswith("llama_"):
        return load_local_based(run_name)
    elif run_name.startswith("llg_"):
        from rule_gen.reddit.classifier_loader.llama_guard_based import load_llama_guard_based
        return load_llama_guard_based(run_name)
    elif run_name.startswith("chatgpt_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_chatgpt_based
        return load_chatgpt_based(run_name)
    elif run_name.startswith("colbert"):
        from rule_gen.reddit.classifier_loader.get_qd_predictor import get_colbert_const
        return get_colbert_const(run_name)
    elif run_name.startswith("col"):
        from rule_gen.reddit.classifier_loader.get_qd_predictor import get_qd_predictor_w_conf
        return get_qd_predictor_w_conf(run_name)
    elif run_name.startswith("proto"):
        from rule_gen.reddit.classifier_loader.proto_predictor import get_proto_predictor
        return get_proto_predictor(run_name)
    elif run_name.startswith("conf_"):
        return load_from_conf(run_name)
    else:
        raise ValueError(f"{run_name} is not expected")


