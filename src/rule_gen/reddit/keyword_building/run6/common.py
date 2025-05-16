import pickle

from rule_gen.reddit.path_helper import get_rp_path


def load_run6_10k_terms(n):
    return load_run6_10k_terms_column(0, n)


def load_run6_10k_terms_column(col_i, n):
    topk_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
    voca = pickle.load(open(topk_path, "rb"))
    term_list = [e[col_i] for e in voca]
    return term_list


def load_run6_term_text_to_term(n):
    topk_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
    voca = pickle.load(open(topk_path, "rb"))
    return {e[1]: e[0] for e in voca}


def load_run6_10k_text(n):
    return load_run6_10k_terms_column(1, n)



def get_bert_basic_tokenizer():
    base_model = 'bert-base-uncased'
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    return tokenize_fn
