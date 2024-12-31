from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_text_list_save_path, get_wrong_pred_save_path, get_csv_dataset_path


def load_fail_entries() -> list[tuple[str, str]]:
    fail_entries = []
    for i in range(10):
        run_name = f"ft12_fold_{i}"
        dataset = f"toxigen_train_fold_{i}"
        save_path: str = get_wrong_pred_save_path(run_name, dataset)
        data_id_list = [e[0] for e in read_csv(save_path)]
        test_dataset: ToxigenBinary = ToxigenBinary("train")
        text_d = {e["id"]: e["text"] for e in test_dataset}
        l = [(data_id, text_d[data_id]) for data_id in data_id_list]
        fail_entries.extend(l)
    return fail_entries


def main():
    save_path = get_text_list_save_path(f"toxigen_train_para_all_fold_selected")
    para_entries = read_csv(save_path)
    fail_entries = load_fail_entries()

    assert len(para_entries) == len(fail_entries)
    out_table = []
    for (data_id, text), para in zip(fail_entries, para_entries):
        out_table.append((data_id, text, para[0]))

    save_path = get_csv_dataset_path("toxigen_fold_fail_para")
    save_csv(out_table, save_path)



if __name__ == "__main__":
    main()