from toxicity.io_helper import read_csv_column, save_text_list_as_csv
from toxicity.llama_helper.lf_client import transform_text_by_llm
from toxicity.path_helper import get_toxigen_failure_save_path, get_text_list_save_path


def main():
    instruction = "Paraphrase the given sentence while keeping its toxic nature."
    p = get_toxigen_failure_save_path("toxigen_train_head_1000", "lg2_2")
    text_list = read_csv_column(p, 1)

    out_itr = transform_text_by_llm(instruction, text_list)

    save_path = get_text_list_save_path("toxigen1000_fail_para_toxic")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()
