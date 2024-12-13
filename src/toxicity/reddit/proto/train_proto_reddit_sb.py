import logging

import fire
from omegaconf import OmegaConf

from toxicity.io_helper import init_logging
from toxicity.reddit.proto.train_proto_reddit import protonet_train_exp

LOG = logging.getLogger(__name__)


def main(
    sb = "churning",
    debug=False,
):
    init_logging()
    conf = OmegaConf.create({
        'dataset_name': f'{sb}',
        'run_name': f'proto3_{sb}',
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'k_protos': 20,
        'learning_rate': 0.005
    })
    LOG.info(str(conf))
    protonet_train_exp(conf, False, debug)


if __name__ == "__main__":
    fire.Fire(main)
