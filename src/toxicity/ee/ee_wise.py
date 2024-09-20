import logging

from taskman_client.wrapper3 import JobContext
from toxicity.ee.easy_edit_main import get_hparams, edit_exp
import fire

LOG = logging.getLogger(__name__)


def main(target_layer):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    LOG.info(__name__)

    with JobContext("wise"):
        hp_path = 'confs/EasyEdit/hparams/WISE/llama_guard2.yaml'
        hparams = get_hparams(hp_path)
        params = hparams.inner_params
        new_params = []
        for t in params:
            new_params.append(t.replace("28", str(target_layer)))
        hparams.inner_params = new_params
        LOG.info("%s", str(hparams))
        run_name = f"ee_wise_l{target_layer}"
        edit_exp(hparams, run_name)


if __name__ == "__main__":
    fire.Fire(main)
