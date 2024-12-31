from desk_util.io_helper import read_csv_column, save_text_list_as_csv
from llama_user.llama_helper.lf_client import transform_text_by_llm
from desk_util.path_helper import get_toxigen_failure_save_path, get_text_list_save_path


def main():
    instruction = "Select one or two words from this text and paraphrase it by replacing it with comparable synonyms."
    p = get_toxigen_failure_save_path("toxigen_train_head_1000", "lg2_2")
    text_list = read_csv_column(p, 1)

    out_list = transform_text_by_llm(instruction, text_list)

    save_path = get_text_list_save_path("toxigen1000_fail_para_failspy")
    save_text_list_as_csv(out_list, save_path)


if __name__ == "__main__":
    main()
