import json
from rule_gen.cpath import output_root_path
import os
import fire


def main(sb= "TwoXChromosomes"):
    ngram_path2 = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_all", f"{sb}.json")
    j = json.load(open(ngram_path2))
    save_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_all_sel", f"{sb}.json")
    output = []
    for e in j:
        text = e["text"]
        subs = e["strong_sub_texts"]
        def get_score(entry):
            n_tokens = len(entry["sub_text"].split())
            return entry["score"] + n_tokens * 0.01
        subs.sort(key=get_score, reverse=True)
        sub_text = subs[0]["sub_text"]
        feature_str = text.replace(sub_text, f"<reason> {sub_text} </reason>")
        output.append(feature_str)

    json.dump(output, open(save_path, "w"), indent=2)


if __name__ == "__main__":
    fire.Fire(main)