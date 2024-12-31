import fire
import torch
import os

import numpy as np

from omegaconf import OmegaConf

from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.classifier_loader.torch_misc import get_device
from rule_gen.reddit.colbert.cb_inf import QDPredictor
from rule_gen.reddit.colbert.modeling import get_arch_class
from rule_gen.reddit.colbert.query_builders import get_sb_to_query

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def compute_stdev(run_name, model_path):
    model_name, sb = run_name.split("/")
    conf_path = os.path.join("confs", "col", f"{model_name}.yaml")
    conf = OmegaConf.load(conf_path)
    if model_path is None:
        model_path = get_model_save_path(conf.run_name)
    sb_to_query = get_sb_to_query(conf.sb_strategy)
    arch_class = get_arch_class(conf.arch_name)
    predictor = QDPredictor(arch_class, model_path, get_device())

    q = sb_to_query(sb)
    d1 = "Fuck compassion Trump supporters are wastes of biomass"
    d2 = "This is very nice idea."
    d2 = "Maybe consider changing the deck."

    with torch.no_grad():
        inputs = predictor.preprocess(q, d1)
        scores = predictor.model.get_2d_scores(**inputs)

    query_input_ids = inputs["query_input_ids"][0].cpu()
    doc_input_ids = inputs["doc_input_ids"][0].cpu()

    def convert(input_ids):
        tokens = predictor.tokenizer.convert_ids_to_tokens(input_ids)
        end = tokens.index("[PAD]")
        return tokens[:end]

    l1 = len(convert(query_input_ids))
    l2 = len(convert(doc_input_ids))
    valid_scores = scores[0, :l1, :l2].cpu().numpy()
    print(valid_scores)
    stdev = np.std(valid_scores)
    print(stdev)


def main(model_name, model_path=None):
    compute_stdev(model_name + "/hearthstone", model_path)


if __name__ == "__main__":
    fire.Fire(main)
