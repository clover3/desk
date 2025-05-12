import os
import pickle

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    voca_d = {}
    for i in [1, 2]:
        voca_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                 "s9_ngram_93_voca", f"{i}.txt")
        print("Reading voca")
        voca = [l.strip() for l in open(voca_path, "r")]
        voca_d[i] = voca

    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list:
        for i in [1, 2]:
            voca = voca_d[i]
            score_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca_score",
                                      f"{sb}.{i}.pkl")
            text_out_path = os.path.join(output_root_path, "reddit", "rule_processing", "s9_ngram_93_voca_score_text",
                                         f"{sb}.{i}.txt")
            make_parent_exists(text_out_path)

            if os.path.exists(score_path):
                scores = pickle.load(open(score_path, "rb"))
                l = list(zip(voca, scores))
                l.sort(key=lambda x: x[1], reverse=True)
                assert len(scores) == len(voca)
                with open(text_out_path, "w") as f:
                    for term, s in l:
                        f.write("{}\t{}\n".format(term, s))


if __name__ == "__main__":
    main()
