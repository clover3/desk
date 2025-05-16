import pickle
from collections import defaultdict

from rule_gen.reddit.keyword_building.run6.common import load_run6_term_text_to_term
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    inv_index_path = get_rp_path("run6_inv_index2_train.pkl")
    inv_index_d = pickle.load(open(inv_index_path, "rb"))

    text_d_path = get_rp_path("run6_doc_map.pkl")
    text_d = pickle.load(open(text_d_path, "rb"))
    print("inv_index_d", inv_index_d.keys())
    term_text_to_term_d = {}
    for n in range(1, 11):
        term_text_to_term_d[n] = load_run6_term_text_to_term(n)


    while True:
        try:
            term = input("Enter a term: ")
            for n in inv_index_d.keys():
                if term in term_text_to_term_d[n]:
                    term = term_text_to_term_d[n][term]
                if term in inv_index_d[n]:
                    postings = inv_index_d[n][term]
                    cnt = 0
                    for doc_id in postings:
                        print(doc_id, text_d[doc_id])
                        print("----")
                        cnt += 1
                        if cnt > 10:
                            break
            print("<---->")
        except (ValueError, KeyError) as e:
            print(e)


if __name__ == "__main__":
    main()
