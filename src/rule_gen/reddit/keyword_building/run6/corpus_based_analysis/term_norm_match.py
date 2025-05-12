from collections import Counter
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import load_jsonl
from rule_gen.reddit.keyword_building.run6.common import get_bert_basic_tokenizer
from rule_gen.reddit.path_helper import get_rp_path


def load_voca_to_doc_id(n):
    mapping_save_path = get_rp_path("run6_voca_doc_map", f"{n}.jsonl")
    make_parent_exists(mapping_save_path)
    items = load_jsonl(mapping_save_path)
    out_d = {}
    for e in items:
        term = e["term"]
        doc_name = e["doc_name"]
        out_d[term] = doc_name

    return out_d


def load_doc_id_to_response():
    model_name = "gpt-4o"
    output_path = get_rp_path(f"run6_unique_docs_why_{model_name}.jsonl")
    items = load_jsonl(output_path)
    out_d = {}
    for e in items:
        doc_name = e["doc_name"]
        out_d[doc_name] = e["text"]
    return out_d


def response_feature_factory():
    tokenize_fn = get_bert_basic_tokenizer()

    def get_feature(text):
        text = text.replace("*", "")
        tokens = tokenize_fn(text)
        bow = Counter(tokens)
        # for seq in ngrams(tokens, 2):
        #     bow[tuple(seq)] += 1
        return bow

    return get_feature
    


def load_doc_id_to_bow():
    out_d = load_doc_id_to_response()
    get_feature = response_feature_factory()
    out_d = {k: get_feature(v) for k, v in out_d.items()}
    return out_d
