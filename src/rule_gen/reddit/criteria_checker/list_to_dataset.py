from desk_util.path_helper import get_csv_dataset_path, get_label_path
from rule_gen.cpath import output_root_path
import os
import json
from desk_util.io_helper import save_csv


def main():
    neg_path = os.path.join(output_root_path, "reddit", "subset", f"mod_pos.csv")
    obj = json.load(open(neg_path))
    dataset_name = "mod_pos"
    payload = []
    labels = []
    for idx, line in enumerate(obj):
        data_id = "{}_{}".format(dataset_name, idx)
        payload.append((data_id, line))
        labels.append((data_id, 1))

    p = get_csv_dataset_path(dataset_name)
    save_csv(payload, p)

    p = get_label_path(dataset_name)
    save_csv(labels, p)


if __name__ == "__main__":
    main()