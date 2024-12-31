import os

from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv


def main():
    dataset = "toxigen_train_fail_para_fold_0"
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    print(payload)


if __name__ == "__main__":
    main()
