from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import save_text_list_as_csv
from toxicity.llama_helper.lf_client import transform_text_by_llm
from toxicity.path_helper import get_text_list_save_path
from toxicity.prompt_edit.prompt_fix_exp import load_predictions_parsed


def main():
    n_item = 100
    preds = load_predictions_parsed("api_1", n_item)
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]
    text_list = [e["text"] for e in test_dataset]
    instruction = "Select one or two words from this text and paraphrase it by replacing it with comparable synonyms."
    out_itr = transform_text_by_llm(instruction, text_list)
    save_path = get_text_list_save_path("toxigen_head_para")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()

