from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import save_text_list_as_csv, read_csv
from llama_user.llama_helper.lf_client import transform_text_by_llm
from desk_util.path_helper import get_text_list_save_path, get_wrong_pred_save_path


def main():
    for i in range(10):
        run_name = f"ft12_fold_{i}"
        dataset = f"toxigen_train_fold_{i}"
        save_path: str = get_wrong_pred_save_path(run_name, dataset)
        data_id_list = [e[0] for e in read_csv(save_path)]
        test_dataset: ToxigenBinary = ToxigenBinary("train")
        text_d = {e["id"]: e["text"] for e in test_dataset}
        text_list = [text_d[data_id] for data_id in data_id_list]
        instruction = "Select one or two words from this text and paraphrase it by replacing it with comparable synonyms."
        out_itr = transform_text_by_llm(instruction, text_list)
        save_path = get_text_list_save_path(f"toxigen_train_para_fold_{i}")
        save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()

