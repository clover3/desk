from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import save_text_list_as_csv
from toxicity.path_helper import get_text_list_save_path


def main():
    n_item = 100
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]
    outputs = [e["text"] for e in test_dataset]
    save_path = get_text_list_save_path("toxigen_head_100")
    save_text_list_as_csv(outputs, save_path)


if __name__ == "__main__":
    main()
