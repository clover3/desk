import os
import pickle
from collections import defaultdict

import unicodedata
from transformers import AutoTokenizer, BertTokenizer
from transformers.models.bert.tokenization_bert import whitespace_tokenize

from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_rp_path


def check_no_post(tokens, acc_norm_text):
    sp_text = " ".join(tokens)
    no_post_list = [">", "\"", "\'"]
    for ch in no_post_list:
        if ch in tokens:
            cand_text = sp_text.replace(ch + " ", ch)
            if cand_text in acc_norm_text:
                return cand_text
    return None


def check_no_pre(tokens, acc_norm_text):
    sp_text = " ".join(tokens)
    no_pre_list = [".", ",", "\"", "\'", "?", "!"]
    for ch in no_pre_list:
        if ch in tokens:
            cand_text = sp_text.replace(" " + ch, ch)
            if cand_text in acc_norm_text:
                return cand_text
    return None


if __name__ == '__main__':
    rev_src_path = get_rp_path("run6_voca_rev_src.pkl")
    rev_map: dict[int, dict[tuple, str]] = pickle.load(open(rev_src_path, "rb"))
    base_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(base_model)

    out_rev_map = defaultdict(dict)
    for n, d in rev_map.items():
        for tokens, sub_text in d.items():
            sub_text_low = sub_text.lower()
            unicode_normalized_text = unicodedata.normalize("NFC", sub_text_low)
            orig_tokens = whitespace_tokenize(unicode_normalized_text)
            space_norm_text = " ".join(orig_tokens)
            acc_norm_text = tokenizer.basic_tokenizer._run_strip_accents(space_norm_text)

            sp_text = " ".join(tokens)
            if sp_text in acc_norm_text:
                out_rev_map[n][tokens] = sp_text
                continue

            nosp_text = "".join(tokens)
            if nosp_text in acc_norm_text:
                out_rev_map[n][tokens] = nosp_text
                continue

            ret = check_no_post(tokens, acc_norm_text)
            if ret is not None:
                out_rev_map[n][tokens] = ret
                continue
            ret = check_no_pre(tokens, acc_norm_text)
            if ret is not None:
                out_rev_map[n][tokens] = ret
                continue

            # out_n += 1
            # print(tokens, sub_text, acc_norm_text)
            # if out_n > 100:
            #     break
        #
        if n > 10:
            break

    save_path = get_rp_path(
                                "run6_voca_rev_src_map.pkl")
    pickle.dump(out_rev_map, open(save_path, "wb"))
