import json
import os
import fire
from rule_gen.cpath import output_root_path


def infer_tokens(sb="TwoXChromosomes"):
    path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_g", f"{sb}.json")
    obj = json.load(open(path, "r"))
    json.dump(obj, open(path, "w"), indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(infer_tokens)
