import fire
import logging

from transformers import BertTokenizer

from toxicity.io_helper import init_logging
from toxicity.reddit.colbert.dataset_builder import TwoColumnDatasetLoader
from toxicity.reddit.colbert.modeling import ColA
from toxicity.reddit.colbert.train_common import get_reddit_data_arguments, get_default_training_argument, \
    train_bert_like_model

LOG = logging.getLogger(__name__)


def main(debug=False, subreddit="TwoXChromosomes"):
    init_logging()
    loader = TwoColumnDatasetLoader(subreddit)
    run_name = f"colbert_{subreddit}"
    dataset_args = get_reddit_data_arguments(debug, subreddit)
    training_args = get_default_training_argument(run_name)
    base_model = 'bert-base-uncased'
    colbert = ColA.from_pretrained(base_model)
    tokenizer = BertTokenizer.from_pretrained(base_model)
    colbert.colbert_set_up(tokenizer)

    train_bert_like_model(
        colbert, tokenizer, training_args, dataset_args,
        loader, run_name, subreddit, debug)


if __name__ == "__main__":
    fire.Fire(main)
