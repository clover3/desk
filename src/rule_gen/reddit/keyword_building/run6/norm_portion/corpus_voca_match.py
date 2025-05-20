import json

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from rule_gen.reddit.keyword_building.run6.common import get_bert_basic_tokenizer, load_run6_10k_terms
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, load_subreddit_list, get_rp_path
from nltk import ngrams


#

def enum_documents(sb, n_item=100):
    save_path = get_reddit_train_data_path_ex("train_data2", sb, "train")
    items = read_csv(save_path)
    items = items[:n_item]

    for idx, (t, l) in enumerate(items):
        doc_id = "{}_{}".format(sb, idx)
        yield doc_id, t, l

    return items


def main():
    # Initialize voca -> cluster mapping
    # Load all voca.
    sb_list = load_subreddit_list()
    tokenize_fn = get_bert_basic_tokenizer()
    n_list = list(range(1, 10))
    voca = set()
    for n in n_list:
        terms = load_run6_10k_terms(n)
        if n == 1:
            assert type(terms[0]) == str
            terms = [(t, ) for t in terms]
        voca.update(terms)
    for sb in sb_list:
        print(sb)
        save_path = get_rp_path("doc_term_match", f"{sb}.json")
        make_parent_exists(save_path)
        output = []
        for doc_id, text, label_s in enum_documents(sb):
            if int(label_s):
                tokens = tokenize_fn(text)
                n = 1
                matched = []
                while n <= len(tokens) and n <= 10:
                    for seq in ngrams(tokens, n):
                        term = tuple(seq)
                        if term in voca:
                            matched.append(term)
                    n += 1
                output.append((doc_id, matched))

        json.dump(output, open(save_path, "w"))

    # for each subreddit
    #   Enum comments
    #       Tokenize, enum n-gram
    #       map to voca
    #
    # Output: list[(doc_id, voca_list)]
    # Next step (doc_id, voca_list) -> doc_id, (cluster, un-mapped voca)
    #   Can this be used to explain generic norm?
    #
    # I want to know which voca frequently appeared but not manually explained


if __name__ == "__main__":
    main()
