from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import save_text_list_as_csv, read_csv_column
from toxicity.llama_helper.lf_client import transform_text_by_llm
from toxicity.path_helper import get_text_list_save_path
from toxicity.prompt_edit.prompt_fix_exp import load_predictions_parsed


def main():
    save_path = get_text_list_save_path("cot_like")
    text_list = read_csv_column(save_path, 0)
    instruction = "If the text was considered toxic, output 'toxic' as a first token. If not, output benign"
    out_itr = transform_text_by_llm(instruction, text_list)
    save_path = get_text_list_save_path("cot_like_parse")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()