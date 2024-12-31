import logging

import fire
from omegaconf import OmegaConf

from desk_util.io_helper import init_logging
from rule_gen.reddit.proto.protory_net2 import ProtoryNet3
from rule_gen.reddit.proto.train_proto_reddit import protonet_train_exp

LOG = logging.getLogger(__name__)


def main(
    sb = "churning",
    k_protos=20,
    learning_rate=0.005,
    epochs=3,
    debug=False,
):
    init_logging()
    conf = OmegaConf.create({
        'dataset_name': f'{sb}',
        'run_name': f'proto_hp_{sb}_{k_protos}_{learning_rate}',
        'base_model_name': 'sentence-transformers/all-MiniLM-L6-v2',
        'k_protos': k_protos,
        'learning_rate': learning_rate,
        'epochs': epochs,
    })
    LOG.info(str(conf))
    protonet_train_exp(ProtoryNet3, conf, False, debug)


if __name__ == "__main__":
    fire.Fire(main)
