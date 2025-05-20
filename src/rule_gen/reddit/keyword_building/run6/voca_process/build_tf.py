import pickle
from collections import Counter

import tqdm
from transformers import BertTokenizer

from rule_gen.reddit.keyword_building.run6.norm_portion.corpus_voca_match import enum_documents
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    base_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    all_tf_df = []
    for sb in tqdm.tqdm(subreddit_list):
        tf = Counter()
        df = Counter()
        for doc, t, l in enum_documents(sb, n_item=10000):
            tokens = tokenize_fn(doc)
            local_tf = Counter(tokens)
            for term, cnt in local_tf.items():
                tf[term] += cnt
                df[term] += 1

        all_tf_df.append((sb, tf, df))

    pkl_path = get_rp_path("all_tf_df.pkl")
    pickle.dump(all_tf_df, open(pkl_path, "wb"))


if __name__ == "__main__":
    main()
