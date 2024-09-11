import os
import fire
from tqdm import tqdm
from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv_column, save_csv


def main():
    dataset = "toxigen_head_100_para_clean"
    src_path: str = os.path.join(output_root_path, "text_list", f"{dataset}.csv")
    text_list = read_csv_column(src_path, 0)
    output = [(str(idx), text) for idx, text in enumerate(text_list)]

    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset}.csv")
    save_csv(output, save_path)


if __name__ == "__main__":
    main()