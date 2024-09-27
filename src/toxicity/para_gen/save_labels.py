from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_wrong_pred_save_path, get_csv_dataset_path, get_label_path


def main():
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    text_d = {e["id"]: e["text"] for e in test_dataset}
    for i in range(10):
        run_name = f"lg2_2"
        dataset = f"toxigen_train_fold_{i}"
        save_path: str = get_wrong_pred_save_path(run_name, dataset)
        wrong_preds = read_csv(save_path)
        output = []
        for data_id, pred, label in wrong_preds:
            output.append((data_id, label))

        save_path = get_label_path(f"toxigen_train_fail_para_fold_{i}")
        save_csv(output, save_path)
        save_path = get_label_path(f"toxigen_train_fail_fold_{i}")
        save_csv(output, save_path)

        output = []
        for data_id, pred, label in wrong_preds:
            output.append((data_id, text_d[data_id]))

        save_path = get_csv_dataset_path(f"toxigen_train_fail_fold_{i}")
        save_csv(output, save_path)


if __name__ == "__main__":
    main()