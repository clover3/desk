import json
import os

import fire

from desk_util.path_helper import get_model_save_path
from rule_gen.cpath import output_root_path
from rule_gen.reddit.bert_probe.probe_inference import ProbeInference
from rule_gen.reddit.bert_probe.probe_sel_top import extract_top_k_important


# Example usage:
def main(sb="askscience", model_name=""):
    print(sb)
    if not model_name:
        model_name = f"bert2_{sb}"
        model_name = model_name + "_probe"
    model_path = get_model_save_path(model_name)

    print("Loading model")
    inf = ProbeInference(model_path)
    j_cluster_path = os.path.join(output_root_path, "clusters", f"bert2_{sb}.json")
    clusters: list[list[str]] = json.load(open(j_cluster_path, "r"))

    j_save_path = os.path.join(output_root_path, "reddit", "clusters_important", f"bert2_{sb}.json")
    print("Extracting top tokens")


    j_out = [extract_top_k_important(inf, c) for c in clusters[:5]]
    json.dump(j_out, open(j_save_path, "w"))



if __name__ == "__main__":
    fire.Fire(main)
