from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from desk_util.io_helper import save_text_list_as_csv
from llama_user.llama_helper.lf_client import transform_text_by_llm
from desk_util.path_helper import get_text_list_save_path


def main():
    n_item = 100
    test_dataset: ToxigenBinary = ToxigenBinary("train")
    test_dataset = list(test_dataset)[:n_item]
    text_list = [e["text"] for e in test_dataset]
    instruction = "Toxicity: "
    instruction += "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
    instruction += "Is this text include stereotypes over certain demographic groups? "
    instruction += "Considering these, generate explanation. The last word should be toxic or benign.\n"
    out_itr = transform_text_by_llm(instruction, text_list)
    save_path = get_text_list_save_path("cot_like")
    save_text_list_as_csv(out_itr, save_path)


if __name__ == "__main__":
    main()

