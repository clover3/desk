import json
from rule_gen.cpath import output_root_path
import os
import fire

from rule_gen.reddit.path_helper import get_rp_path


def main(sb= "TwoXChromosomes"):
    ngram_path2 = get_rp_path("ngram_93_all", f"{sb}.json")

    j = json.load(open(ngram_path2))
    save_path = get_rp_path("ngram_93_all_sub_sel", f"{sb}.json")
    output = []
    for e in j:
        text = e["text"]
        subs = e["strong_sub_texts"]
        def get_score(entry):
            n_tokens = len(entry["sub_text"].split())
            return entry["score"] + n_tokens * 0.01
        subs.sort(key=get_score, reverse=True)
        sub_text = subs[0]["sub_text"]
        output.append({
            "text": text,
            "sub_text": sub_text
        })

    json.dump(output, open(save_path, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)