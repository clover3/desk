import pickle
from collections import Counter

from rule_gen.reddit.keyword_building.run6.score_common.score_loaders import load_mat_terms
from rule_gen.reddit.path_helper import get_rp_path


def print_top_k(counter_list, k=3):
    for label, counter in counter_list:

        print(f"Top-{k} elements for {label}:")
        for term, count in counter.most_common(k):
            print(f"  {term}: {count}")
        print()

    # Create a summed counter
    summed_counter = Counter()
    for _, counter in counter_list:
        summed_counter.update(counter)

    # Print top-k for the summed counter
    print(f"Top-{k} elements for summed counter:")
    for term, count in summed_counter.most_common(k):
        print(f"  {term}: {count}")


def main():

    doc_term_match_path = get_rp_path("clustering", f"100_annot_missing.pkl")
    counter_list = pickle.load(open(doc_term_match_path, "rb"))


    n_list = list(range(1, 10))
    score_mat, term_list, valid_sb_list = load_mat_terms(n_list)

    new_counter_list = []
    for sb, counter in counter_list:
        try:
            sb_idx = valid_sb_list.index(sb)
        except ValueError:
            continue
        new_counter = Counter()
        for term, cnt in counter.items():
            if len(term) == 1:
                idx = term_list.index(term[0])
                continue
            elif len(term) >= 10:
                continue
            else:
                idx = term_list.index(term)
            score = score_mat[idx, sb_idx]
            if score > 0.7:
                new_counter[term] = cnt
        new_counter_list.append((sb, new_counter))

    print_top_k(new_counter_list)



if __name__ == "__main__":
    main()
