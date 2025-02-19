import json
import os

from rule_gen.cpath import output_root_path
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted
from typing import List, Tuple

from chair.misc_lib import make_parent_exists


def main():
    todo = [
        1, 5, 10, 50, 100, 200, 500, 1000, 2000, 4000, 8000
    ]
    todo = [100 * 1000]
    for n_item in todo:
        source: List[Tuple[str, str]] = load_toxigen_formatted(n_item=n_item)

        # Save to json with
        output = []
        for inst, label in source:
            j = {"instruction": inst,
                 "input": "",
                 "output": label
            }
            output.append(j)

        file_name: str = f"toxigen_{n_item}.json"
        save_path: str = os.path.join(output_root_path, "payload", "llamaguard", file_name)
        make_parent_exists(save_path)

        json.dump(output, open(save_path, "w"), indent=2)


if __name__ == "__main__":
    main()