import pickle
from collections import Counter, defaultdict

from rule_gen.reddit.keyword_building.run6.corpus_based_analysis.term_norm_match import load_doc_id_to_response

import re

from rule_gen.reddit.path_helper import get_rp_path


def extract_asterisk():
    print("init")
    doc_id_to_res = load_doc_id_to_response()

    def normalize(text):
        t = text.strip()
        if t[-1] == ":":
            t = t[:-1]
        return t

    regex = r"\*\*([^\n]{1,30}?)\*\*"

    n_doc_matched = 0
    counter= Counter()
    for doc_id, text in doc_id_to_res.items():
        itr = re.findall(regex, text)
        f_is_match = False
        for item in itr:
            term = normalize(item)
            counter[term] += 1
            f_is_match = True

        if f_is_match:
            n_doc_matched += 1

    norm_voca = list(counter.keys())
    save_path = get_rp_path(f"norm_voca", "asterisk_terms.pkl")
    pickle.dump(norm_voca, open(save_path, "wb"))


def do_tokenize_ngram():
    path = get_rp_path(f"norm_voca", "asterisk_terms.pkl")
    src_voca = pickle.load(open(path, "rb"))

    post_voca = set()
    for term in src_voca:
        tokens = re.split(r"[ /]", term)
        post_voca.update(tokens)

    print("Tokenize get {} terms".format(len(post_voca)))
    path = get_rp_path(f"norm_voca", "unigram.pkl")
    pickle.dump(list(post_voca), open(path, "wb"))






def show_uni():
    path = get_rp_path(f"norm_voca", "unigram.pkl")
    uni_voca = pickle.load(open(path, "rb"))
    save_path = get_rp_path(f"norm_voca", "unigram.txt")
    uni_voca.sort()
    with open(save_path, "w") as f:
        for t in uni_voca:
            f.write(f"{t}\n")



def show_asterisk_terms():
    path = get_rp_path(f"norm_voca", "asterisk_terms.pkl")
    uni_voca = pickle.load(open(path, "rb"))
    save_path = get_rp_path(f"norm_voca", "asterisk_terms.txt")
    uni_voca.sort()
    with open(save_path, "w") as f:
        for t in uni_voca:
            f.write(f"{t}\n")


def voca_eval():
    path = get_rp_path(f"norm_voca", "unigram.pkl")
    uni_voca = pickle.load(open(path, "rb"))
    path = get_rp_path(f"norm_voca", "asterisk_terms.pkl")
    ask_voca = pickle.load(open(path, "rb"))
    doc_id_to_res = load_doc_id_to_response()
    voca = ask_voca + uni_voca

    voca = {v.lower() for v in voca}
    counter= Counter()
    for doc_id, text in doc_id_to_res.items():
        l_text = text.lower()
        for term in voca:
            if term in l_text:
                counter[term] += 1

    output_list = list(counter.most_common(600))
    output_list = output_list[300:]
    for term, cnt in output_list:
        print(term, cnt)




if __name__ == "__main__":
    voca_eval()