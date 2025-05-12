import random
from typing import Callable

from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.table_scorer import get_table_clf


def get_random_classifier():
    def predict(text):
        pred = random.randint(0, 1)
        ret = int(pred)
        return ret, 0
    return predict


def get_random_rate_classifier(name):
    _, number_s = name.split("_")
    rate = float("0.{}".format(number_s))
    def predict(text):
        pred = random.random() < rate
        ret = int(pred)
        return ret, 0
    return predict


def get_always_one_clf():
    def predict(text):
        return 1, 0
    return predict


def get_classifier(run_name) -> Callable[[str], tuple[int, float]]:
    if run_name.startswith('bert_ts_'):
        from rule_gen.reddit.bert_pat.pat_classifier import get_pat_predictor
        return get_pat_predictor(run_name)
    elif run_name.startswith('bert_c'):
        from rule_gen.reddit.bert_c.inference import get_bert_c_predictor_by_run_name
        return get_bert_c_predictor_by_run_name(run_name)
    elif run_name.startswith("bert"):
        from rule_gen.reddit.classifier_loader.get_pipeline import get_classifier_pipeline
        return get_classifier_pipeline(run_name)
    elif run_name.startswith("random_"):
        return get_random_rate_classifier(run_name)
    elif run_name == "random":
        return get_random_classifier()
    elif run_name == "always_one":
        return get_always_one_clf()
    elif run_name.startswith("table_"):
        return get_table_clf(run_name)
    elif run_name.startswith("dummy_"):
        from rule_gen.reddit.classifier_loader.prompt_based import dummy_counter
        return dummy_counter(run_name)
    elif run_name.startswith("api_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_api_based
        return load_api_based(run_name)
    elif run_name.startswith("api2_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_api_based2
        return load_api_based2(run_name)
    elif run_name.startswith("llama_s9"):
        from rule_gen.reddit.classifier_loader.s9 import get_s9_classifiers
        return get_s9_classifiers(run_name)
    elif run_name.startswith("llama_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_local_based
        return load_local_based(run_name)
    elif run_name.startswith("llg_"):
        from rule_gen.reddit.classifier_loader.llama_guard_based import load_llama_guard_based
        return load_llama_guard_based(run_name)
    elif run_name.startswith("lf"):
        from rule_gen.reddit.llama.load_llama_inst import get_lf_predictor_w_conf
        return get_lf_predictor_w_conf(run_name)
    elif run_name.startswith("chatgpt_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_chatgpt_based
        return load_chatgpt_based(run_name)
    elif run_name.startswith("colbert"):
        from rule_gen.reddit.classifier_loader.get_qd_predictor import get_colbert_const
        return get_colbert_const(run_name)
    elif run_name.startswith("col"):
        from rule_gen.reddit.classifier_loader.get_qd_predictor import get_qd_predictor_w_conf
        return get_qd_predictor_w_conf(run_name)
    elif run_name.startswith("ce_"):
        from rule_gen.reddit.classifier_loader.get_qd_predictor import get_qd_predictor_w_conf
        from rule_gen.reddit.base_bert.concat_bert_inf import get_ce_predictor_w_conf
        return get_ce_predictor_w_conf(run_name)
    elif run_name.startswith("proto"):
        from rule_gen.reddit.classifier_loader.proto_predictor import get_proto_predictor
        return get_proto_predictor(run_name)
    elif run_name.startswith("conf_"):
        from rule_gen.reddit.classifier_loader.prompt_based import load_from_conf
        return load_from_conf(run_name)
    else:
        raise ValueError(f"{run_name} is not expected")


