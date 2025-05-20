import fire

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_pat.infer_tokens import PatInferenceFirst


def main(sb="TwoXChromosomes"):
    model_name = f"bert_ts_{sb}"
    pat = PatInferenceFirst(get_model_save_path(model_name))
    while True:
        text = input("Enter your text: ")
        ret = pat.get_full_text_score(text)
        print(ret)


if __name__ == "__main__":
    fire.Fire(main)
