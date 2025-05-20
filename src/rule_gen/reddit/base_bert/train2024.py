import logging

import fire

from desk_util.io_helper import init_logging
from rule_gen.reddit.base_bert.train2 import train_subreddit_classifier_inner

LOG = logging.getLogger(__name__)


def train_sb_classifier(sb="askscience_head"):
    init_logging()
    data_name = "2024_2"
    model_name = f"bert2024_{sb}"

    train_subreddit_classifier_inner(data_name, model_name, sb)


# Example usage:
if __name__ == "__main__":
    fire.Fire(train_sb_classifier)
