import pickle

import fire

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.run6.common import load_run6_10k_text
from rule_gen.reddit.path_helper import get_rp_path


def main():
    metric = "mean"
    score_dir_name = "run6_10k_score"

    for n in range(1, 10):
        voca = load_run6_10k_text(n)
        score_path = get_rp_path(score_dir_name, f"{metric}.{n}.pkl")
        scores = pickle.load(open(score_path, "rb"))

        l = list(zip(voca, scores))
        l.sort(key=lambda x: x[1], reverse=True)
        assert len(scores) == len(voca)
        text_out_path = get_rp_path(f"{score_dir_name}", f"{metric}.{n}.txt")
        make_parent_exists(text_out_path)

        with open(text_out_path, "w") as f:
            for term, s in l:
                f.write("{}\t{}\n".format(term, s))


if __name__ == "__main__":
    fire.Fire(main)
