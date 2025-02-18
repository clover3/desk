
import fire

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.bert_pat.pat_modeling import CombineByScoreAdd, BertPatFirst


def main(sb= "TwoXChromosomes"):
    model_name = f"bert_ts_{sb}"

    model_path = get_model_save_path(model_name)
    model = BertPatFirst.from_pretrained(
        model_path,
        combine_layer_factory=CombineByScoreAdd
    )
    print("Model loaded")


if __name__ == "__main__":
    fire.Fire(main)