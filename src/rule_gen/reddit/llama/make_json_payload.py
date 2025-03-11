import json
import shutil
from pathlib import Path
from chair.misc_lib import make_parent_exists
from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
from rule_gen.reddit.llama.prompt_helper import get_prompt_factory_both_rule, get_prompt_fn_from_type
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex



def register_dataset(dataset_path, dataset_name):
    dataset_info_path = "/sfs/gpfs/tardis/home/qdb5sn/work/LLaMA-Factory/data/dataset_info.json"
    file_name = f"{dataset_name}.json"
    with open(dataset_info_path, "r") as dataset_info_file:
        dataset_info = json.load(dataset_info_file)

    dataset_info[dataset_name] = {"filename": file_name}
    with open(dataset_info_path, "w") as dataset_info_file:
        json.dump(dataset_info, dataset_info_file, indent=4)

    dst = Path("/sfs/gpfs/tardis/home/qdb5sn/work/LLaMA-Factory/data") / file_name
    shutil.copyfile(dataset_path, dst)


def save_json_payload(src_data_name, save_data_name, get_prompt):
    label_mapping = {1: "yes", 0: "no"}
    src_data_path = get_reddit_train_data_path_ex("train_data2", src_data_name, "train")
    data = read_csv(src_data_path)
    save_path = Path(output_root_path) / "reddit" / "lf_data" / "{}.json".format(save_data_name)
    save_data = []
    for sb, text, label in data:
        prompt = get_prompt(text, sb)
        e = {
            "instruction": prompt,
            "input": "",
            "output": label_mapping[int(label)]
        }
        save_data.append(e)

    make_parent_exists(save_path)
    json.dump(save_data, open(save_path, "w"), indent=4)
    register_dataset(save_path, save_data_name)


def main():
    src_data_name = "train_comb3"
    prompt_type = "both_rule"
    get_prompt = get_prompt_fn_from_type(prompt_type)
    save_data_name = "both_rule"
    save_json_payload(src_data_name, save_data_name, get_prompt)


def main2():
    src_data_name = "train_comb3"
    prompt_type = "sb_name"
    get_prompt = get_prompt_fn_from_type(prompt_type)
    save_data_name = "sb_name2"
    save_json_payload(src_data_name, save_data_name, get_prompt)


if __name__ == "__main__":
    main2()
