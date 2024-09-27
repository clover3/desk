from collections import defaultdict
from toxicity.dataset_helper.load_toxigen import ToxigenBinary

from toxicity.io_helper import read_csv_column, save_text_list_as_csv, read_csv, save_csv
from toxicity.path_helper import get_wrong_pred_save_path, \
    get_text_list_save_path, load_csv_dataset, get_csv_dataset_path


def main():
    s = get_text_list_save_path("ids_to_gen")
    ids = read_csv_column(s, 0)
    save_path = get_text_list_save_path(f"toxigen_train_para_all_fold_selected_v2_clean")
    paras = read_csv_column(save_path, 0)

    assert len(paras) == len(ids)
    para_d = dict(zip(ids, paras))

    save_path: str = get_wrong_pred_save_path("lg2_2", "toxigen_train_fold_all")
    ids_todo = read_csv_column(save_path, 0)
    para_d2 = {r[0]: r[2] for r in load_csv_dataset("toxigen_fold_fail_para_clean_v1")}


    per_fold_para = defaultdict(list)
    per_fold_fail = defaultdict(list)
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    text_d = {e["id"]: e["text"] for e in test_dataset}

    for data_id in ids_todo:
        try:
            para_text = para_d[data_id]
        except KeyError:
            para_text = para_d2[data_id]
        fold_idx = int(data_id) // 100
        per_fold_para[fold_idx].append((data_id, para_text))
        per_fold_fail[fold_idx].append((data_id, text_d[data_id]))

    for i in range(10):
        dataset_name = f"toxigen_train_fail_para_fold_{i}"
        save_path = get_csv_dataset_path(dataset_name)
        save_csv(per_fold_para[i], save_path)

        dataset_name = f"toxigen_train_fail_fold_{i}"
        save_path = get_csv_dataset_path(dataset_name)
        save_csv(per_fold_fail[i], save_path)


if __name__ == "__main__":
    main()
