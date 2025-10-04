import logging
import os

import fire

from desk_util.io_helper import init_logging
from desk_util.path_helper import get_model_save_path, get_model_log_save_dir_path
from rule_gen.reddit.base_bert.reddit_train_bert import build_training_argument, DataArguments, finetune_bert
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex

LOG = logging.getLogger(__name__)


def train_subreddit_classifier(sb="askscience_head"):
    init_logging()
    data_name = "train_data2"
    model_name = f"bert2_{sb}"
    base_model = 'bert-base-uncased'

    output_dir = get_model_save_path(model_name)
    final_model_dir = get_model_save_path(model_name)
    sf_path = os.path.join(output_dir, "model.safetensors")
    if os.path.exists(sf_path):
        print("Model exists. Skip training")
    logging_dir = get_model_log_save_dir_path(model_name)
    max_length = 256
    training_args = build_training_argument(logging_dir, output_dir)
    dataset_args = DataArguments(
        train_data_path=get_reddit_train_data_path_ex(data_name, sb, "train"),
        eval_data_path=get_reddit_train_data_path_ex(data_name, sb, "val"),
        max_length=max_length
    )

    eval_result = finetune_bert(
        model_name=base_model,
        training_args=training_args,
        dataset_args=dataset_args,
        final_model_dir=final_model_dir,
    )

# Example usage:
if __name__ == "__main__":
    fire.Fire(train_subreddit_classifier)
