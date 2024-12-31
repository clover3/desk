import json
import os

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from toxicity.llama_guard.llama_guard_formatter import LlamaGuardFormatter
from desk_util.path_helper import get_csv_dataset_path, get_label_path


def gen_fold():
    formatter = LlamaGuardFormatter()
    split = "train"
    for fold_idx in range(0, 10):
        save_path = get_csv_dataset_path(f"toxigen_{split}_fold_{fold_idx}")
        payload = read_csv(save_path)
        save_path = get_label_path(f"toxigen_{split}_fold_{fold_idx}")
        label_d: dict[str, str] = dict(read_csv(save_path))
        out_data = []
        for data_id, text in payload:
            prompt = formatter.get_prompt([text])
            label = formatter.get_label_str(int(label_d[data_id]))
            j = {"instruction": prompt,
                 "input": "",
                 "output": label
                 }
            out_data.append(j)

        file_name: str = f"toxigen_fold_{fold_idx}.json"
        save_path: str = os.path.join(output_root_path, "payload", "llamaguard", file_name)
        make_parent_exists(save_path)

        json.dump(out_data, open(save_path, "w"), indent=2)


def main():
    gen_fold()


if __name__ == "__main__":
    main()
