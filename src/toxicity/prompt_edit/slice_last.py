from desk_util.io_helper import save_text_list_as_csv, read_csv_column
from llama_user.llama_helper.lf_client import transform_text_by_llm
from desk_util.path_helper import get_text_list_save_path


def main():
    save_path = get_text_list_save_path("cot_like")
    text_list = read_csv_column(save_path, 0)
    instruction = "If the text was considered toxic, output 'toxic' as a first token. If not, output benign"
    out_itr = transform_text_by_llm(instruction, text_list)
    save_path = get_text_list_save_path("cot_like_parse")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()