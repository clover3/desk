from collections import defaultdict

from desk_util.io_helper import save_csv
from desk_util.path_helper import load_csv_dataset, get_csv_dataset_path


def main():
    src = load_csv_dataset("toxigen_fold_fail_para_clean")

    per_fold = defaultdict(list)
    for data_id, orig, para in src:
        fold_idx = int(data_id) // 100
        per_fold[fold_idx].append((data_id, para))

    for i in range(10):
        dataset_name = f"toxigen_train_fail_para_fold_{i}"
        save_path = get_csv_dataset_path(dataset_name)
        save_csv(per_fold[i], save_path)



if __name__ == "__main__":
    main()