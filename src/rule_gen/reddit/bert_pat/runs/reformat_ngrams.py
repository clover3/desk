import logging
import os
import fire

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.bert_pat.ngram_common import load_ngram_outputs_grouped

LOG = logging.getLogger(__name__)

import json


def main(sb="TwoXChromosomes"):
    ngram_path = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93", f"{sb}.json")
    j = load_ngram_outputs_grouped(ngram_path)
    ngram_path2 = os.path.join(output_root_path, "reddit", "rule_processing", "ngram_93_g", f"{sb}.json")
    make_parent_exists(ngram_path2)
    json.dump(j, open(ngram_path2, "w"), indent=2)



if __name__ == "__main__":
    fire.Fire(main)
