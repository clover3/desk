import json
from pathlib import Path

from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.llama.lf_util import register_dataset


def make_register_reddit_prompts_for_lf(get_prompt, save_data_name, src_data_path):
    save_path = Path(output_root_path) / "reddit" / "lf_data" / "{}.json".format(save_data_name)
    save_data = []
    label_mapping = {1: "yes", 0: "no"}
    data = read_csv(src_data_path)
    for sb, text, label in data:
        e = {
            "instruction": "",
            "input": get_prompt(text, sb),
            "output": label_mapping[int(label)]
        }
        save_data.append(e)
    make_parent_exists(save_path)
    json.dump(save_data, open(save_path, "w"), indent=4)
    register_dataset(save_path, save_data_name)
