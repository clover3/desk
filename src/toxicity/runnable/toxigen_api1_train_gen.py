import json
import os

from toxicity.cpath import output_root_path
from toxicity.dataset_helper.load_toxigen import ToxigenBinary

from chair.misc_lib import make_parent_exists
from toxicity.predictors.api_based import get_api_1_prompt


def main():
    head_inst = get_api_1_prompt()
    todo = [
        1, 10, 100, 1000, 100 * 1000
    ]
    split = "train"
    ds = list(ToxigenBinary(split))

    for n_item in todo:
        print("N_item = ", n_item)
        data = ds[:n_item]

        # Save to json with
        output = []
        for e in data:
            label = ["benign", "toxic"][e["label"]]
            inst = f"{head_inst}\n Text: {e['text']}"
            j = {"instruction": inst,
                 "input": "",
                 "output": label
            }
            output.append(j)
        file_name: str = f"toxigen_{n_item}.json"
        save_path: str = os.path.join(
            output_root_path, "payload", "api_1", file_name)
        make_parent_exists(save_path)
        json.dump(output, open(save_path, "w"), indent=2)


if __name__ == "__main__":
    main()