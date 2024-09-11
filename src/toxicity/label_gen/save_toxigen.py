from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import save_csv
from toxicity.path_helper import get_label_path, get_csv_dataset_path


def main():
    for split in {"train", "test"}:
        test_dataset: ToxigenBinary = ToxigenBinary(split)
        labels = [(e['id'], e['text']) for e in test_dataset]
        save_path = get_csv_dataset_path(f"toxigen_{split}")
        save_csv(labels, save_path)


def gen_head():
    n = 100
    for split in {"train", "test"}:
        test_dataset: ToxigenBinary = ToxigenBinary(split)
        labels = [(e['id'], e['text']) for e in test_dataset]
        labels = labels[:n]
        save_path = get_csv_dataset_path(f"toxigen_{split}_head_{n}")
        save_csv(labels, save_path)




if __name__ == "__main__":
    gen_head()
    main()
