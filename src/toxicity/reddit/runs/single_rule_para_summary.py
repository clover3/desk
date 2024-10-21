import math
import os

from chair.misc_lib import make_parent_exists
from toxicity.cpath import output_root_path
from toxicity.io_helper import save_csv
from toxicity.path_helper import load_clf_pred
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.reddit.runs.single_rule_summary import single_run_result
from toxicity.runnable.run_eval import load_labels, clf_eval


def single_rule_result(rule_idx, rule_sb):
    run_name = f"api_srr_{rule_sb}_{rule_idx}_detail"
    single_run_result(run_name)

def main():
    rule_sb = "TwoXChromosomes"
    rule_idx = 0
    single_rule_result(rule_idx, rule_sb)


if __name__ == "__main__":
    main()
