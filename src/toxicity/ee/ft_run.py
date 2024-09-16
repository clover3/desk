import logging
from toxicity.ee.easy_edit_main import get_hparams, edit_exp
LOG = logging.getLogger(__name__)
import fire


def main(target_layer: int):
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    LOG.info(__name__)
    hp_path = 'confs/EasyEdit/hparams/FT/llama_guard2.yaml'
    hparams = get_hparams(hp_path)
    hparams.layers = [target_layer]
    LOG.info("%s", str(hparams))
    run_name = f"ee_ft_eow_l{target_layer}"
    edit_exp(hparams, run_name)



if __name__ == "__main__":
    fire.Fire(main)
