import pandas as pd
import logging
import os

import fire
from omegaconf import OmegaConf
from transformers import BertForSequenceClassification
from desk_util.io_helper import init_logging
from rule_gen.reddit.base_bert.text_concat import text_concat_exp
from rule_gen.reddit.colbert.query_builders import get_sb_to_query
from rule_gen.runner.predict_split import predict_sb_split
from rule_gen.reddit.train_common import DatasetLoader
from datasets import Dataset as HFDataset

LOG = logging.getLogger(__name__)


class ThreeColumnDatasetLoader(DatasetLoader):
    def __init__(self, sb_to_query):
        self.sb_to_query = sb_to_query

    def get(self, data_path, max_samples=None):
        df = pd.read_csv(
            data_path,
            na_filter=False,
            keep_default_na=False,
            header=None,
            names=['query', 'document', 'label'],
            dtype={"query": str, "document": str, 'label': int}
        )
        if max_samples is not None:
            df = df.head(max_samples)
        return HFDataset.from_pandas(df)


def main(
        debug=False,
        run_name="",
        do_sb_eval=False,
):
    init_logging()
    conf_path = os.path.join("confs", "cross_encoder", f"{run_name}.yaml")
    conf = OmegaConf.load(conf_path)
    print(conf)

    sb_to_query = get_sb_to_query(conf.sb_strategy)
    builder = ThreeColumnDatasetLoader(sb_to_query)
    arch_class = BertForSequenceClassification
    text_concat_exp(builder, debug, arch_class, conf.run_name, conf.dataset_name)
    if do_sb_eval:
        predict_sb_split(run_name + "/{}", "val")


if __name__ == "__main__":
    fire.Fire(main)
