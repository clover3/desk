import json
from collections import defaultdict

from nltk import ngrams

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.cpath import output_root_path
import os



def main():
    subreddit_list = get_split_subreddit_list("train")

    #
    #   load subtext
    #   dict[word, list[str]]
    #   PPP

    voca = defaultdict(set)

    for sb in subreddit_list:
        print(sb)
        save_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93", f"{sb}.json")
        if not os.path.exists(save_path):
            continue

        items = json.load(open(save_path))
        for res in items:
            for k in res["strong_sub_texts"]:
                sub_text = k["sub_text"]
                tokens = sub_text.split()
                for t in tokens:
                    voca[1].add(t)

                for bi in ngrams(tokens, 2):
                    voca[2].add(bi)


    for key in voca:
        save_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca", f"{key}.txt")
        make_parent_exists(save_path)
        f = open(save_path, "w")
        for v in voca[key]:
            f.write("{}\n".format(v))
        print(key, len(voca[key]))


if __name__ == "__main__":
    main()