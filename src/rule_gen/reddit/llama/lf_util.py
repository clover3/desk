import json
import shutil
from pathlib import Path


def register_dataset(dataset_path, dataset_name):
    dataset_info_path = "/sfs/gpfs/tardis/home/qdb5sn/work/LLaMA-Factory/data/dataset_info.json"
    file_name = f"{dataset_name}.json"
    with open(dataset_info_path, "r") as dataset_info_file:
        dataset_info = json.load(dataset_info_file)

    dataset_info[dataset_name] = {"file_name": file_name}
    with open(dataset_info_path, "w") as dataset_info_file:
        json.dump(dataset_info, dataset_info_file, indent=4)

    dst = Path("/sfs/gpfs/tardis/home/qdb5sn/work/LLaMA-Factory/data") / file_name
    shutil.copyfile(dataset_path, dst)
