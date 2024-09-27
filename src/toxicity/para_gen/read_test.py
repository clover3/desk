import os

from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv


def main():
    dataset = "toxigen_train_fail_para_fold_0"
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    payload = read_csv(save_path)
    print(payload)


if __name__ == "__main__":
    main()
