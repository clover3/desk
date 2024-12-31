from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import save_text_list_as_csv, read_csv_column
from llama_user.llama_helper.lf_client import transform_text_by_llm
from desk_util.path_helper import get_text_list_save_path


def main():
    s = get_text_list_save_path("ids_to_gen")
    data_id_list = read_csv_column(s, 0)
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    text_d = {e["id"]: e["text"] for e in test_dataset}
    text_list = [text_d[data_id] for data_id in data_id_list]
    instruction = "Select one or two words from this text and paraphrase it by replacing it with comparable synonyms."
    out_itr = transform_text_by_llm(instruction, text_list)
    save_path = get_text_list_save_path(f"para_from_ids_to_gen")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()

