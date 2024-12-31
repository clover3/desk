import os
import pickle
from collections import Counter

from chair.misc_lib import TELI, make_parent_exists
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_jsonl
from rule_gen.reddit.corpus_sim.cache_stemmer import CacheStemmer
from rule_gen.reddit.path_helper import load_subreddit_list


def build_tf(text_iter):
    stemmer = CacheStemmer()
    c = Counter()
    for text in text_iter:
        tokens = text.split()
        for t in tokens:
            c[stemmer.stem(t)] += 1
    return c


def load_corpus(sub_reddit):
    load_dir = os.path.join(output_root_path, "reddit", "subreddit_samples")
    file_path = os.path.join(load_dir, sub_reddit + ".jsonl")
    items = read_jsonl(file_path)
    texts = [j['body'] for j in items]
    return texts


def run_build_for(sub_reddit):
    n = 1000 * 1000
    texts = load_corpus(sub_reddit)
    print(len(texts), "texts")
    return build_tf(TELI(texts, len(texts)))


def main():
    for sb in load_subreddit_list():
        print(sb)
        counter = run_build_for(sb)
        save_path = os.path.join(output_root_path, "reddit", "tf", f"{sb}.pkl")
        make_parent_exists(save_path)
        pickle.dump(counter, open(save_path, "wb"))


if __name__ == "__main__":
    main()
