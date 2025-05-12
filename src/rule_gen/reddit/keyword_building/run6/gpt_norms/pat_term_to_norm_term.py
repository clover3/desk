import pickle
from collections import Counter, defaultdict

from desk_util.io_helper import load_jsonl, save_csv
from rule_gen.reddit.path_helper import get_rp_path


def run_with(norm_match_file_name, match_save_dir):
    path = get_rp_path(f"norm_voca", norm_match_file_name)
    doc_id_to_bow_norm: dict[str, Counter] = dict(pickle.load(open(path, "rb")))
    for n in range(1, 10):
        print("{} gram".format(n))
        topk_path = get_rp_path("run6_voca_lm_prob_10k", f"{n}.pkl")
        voca: list = pickle.load(open(topk_path, "rb"))

        term_text_to_key = {}
        for term_key, term_text, _ in voca:
            term_text_to_key[term_text] = term_key

        mapping_save_path = get_rp_path("run6_voca_doc_map", f"{n}.jsonl")
        jsonl = load_jsonl(mapping_save_path)
        output = []
        text_output = []
        for j in jsonl:
            term_text = j['term']
            doc_name = j["doc_name"]
            bow = doc_id_to_bow_norm[doc_name]
            term_key = term_text_to_key[term_text]
            out_row = [term_key, bow]
            output.append(out_row)
            text_output.append((term_text, bow))

        save_path = get_rp_path(match_save_dir, f"{n}.pkl")
        pickle.dump(output, open(save_path, "wb"))
        save_path = get_rp_path(match_save_dir, f"{n}.txt")
        save_csv(text_output, save_path)


def main1():
    norm_match_file_name = "doc_id_bow_norm.pkl"
    match_save_dir = "run6_voca_to_norm"
    run_with(norm_match_file_name, match_save_dir)


def main2():
    norm_match_file_name = "doc_id_bow_man_norm.pkl"
    match_save_dir = "run6_voca_to_man_norm"
    run_with(norm_match_file_name, match_save_dir)


if __name__ == "__main__":
    main2()