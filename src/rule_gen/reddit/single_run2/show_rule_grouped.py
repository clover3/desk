import json

import fire

from chair.misc_lib import group_by
from rule_gen.cpath import output_root_path
import os
from rule_gen.reddit.path_helper import get_reddit_rule_path


def main(sb="fantasyfootball"):
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "rule_sel_gpt_res", f"{sb}.json")
    items = json.load(open(res_save_path, "r"))
    grouped = group_by(items, lambda x: int(x["response"]))
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))

    for g_idx, entries in grouped.items():
        rule_j = rules[g_idx-1]

        if str(g_idx) not in rule_j["summary"]:
            raise ValueError()
        print(json.dumps(rule_j, indent=2))

        for item in entries:
            print(item["text"])
            print("----")




if __name__ == "__main__":
    fire.Fire(main)
