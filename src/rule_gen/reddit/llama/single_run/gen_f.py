import json
from pathlib import Path
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from rule_gen.reddit.llama.lf_util import register_dataset
from rule_gen.reddit.llama.prompt_helper import get_prompt_factory, get_pattern_f_prompt_fn
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex
from rule_gen.cpath import output_root_path


def main():
    src_data_name = "train_comb4"
    save_data_name = "lf_pattern_f"
    save_path = Path(output_root_path) / "reddit" / "lf_data" / "{}.json".format(save_data_name)

    save_data = []
    label_mapping = {1: "yes", 0: "no"}
    src_data_path = get_reddit_train_data_path_ex("train_data2", src_data_name, "train")
    get_prompt = get_pattern_f_prompt_fn()
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


if __name__ == "__main__":
    main()
