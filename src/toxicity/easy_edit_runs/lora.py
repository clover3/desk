import logging
import sys

from easyeditor.editors.edit_runner import EasyEditEditor
from toxicity.ee.easy_edit_main import get_hparams
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_comparison_save_path
from toxicity.ee.edit_exp_runner import EditExperimentRunner
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted, load_toxigen_formatted_inner
import wandb

LOG = logging.getLogger(__name__)


def load_wrong_ids():
    path = get_comparison_save_path("llama_guard2_prompt", "toxigen_train_head_1000")
    entries = read_csv(path)[1:]
    ids = [int(e[0]) for e in entries]
    return ids


def load_toxigen_wrong(n_item):
    data = load_toxigen_formatted_inner("train", "annotated")
    ids = load_wrong_ids()

    wrong_items = []
    for idx, item in enumerate(data):
        if idx in ids:
            wrong_items.append(item)
    return wrong_items[:n_item]


def set_up_wandb(exp_conf):
    wandb.init(config=exp_conf, project="EasyEdit-toxicity")
    base_name = "Lora_WrongOnly_2"
    run_name = f"{base_name}"
    try:
        wandb.run.name = run_name
    except:
        pass
    # wandb.config = hparam


def llama_guard_edit():
    if len(sys.argv) > 1:
        hp_path = sys.argv[1]
    else:
        hp_path = 'confs/EasyEdit/hparams/LoRA/llama_guard2.yaml'
    hparams = get_hparams(hp_path)
    exp_conf = {
        "num_steps": 20,
        "n_edit": 100,
        "n_valid": 50
    }
    hparams.num_steps = exp_conf["num_steps"]
    set_up_wandb(exp_conf)
    edit_payload = load_toxigen_wrong(n_item=exp_conf['n_edit'])
    test_data = load_toxigen_formatted(split="test", n_item=exp_conf['n_valid'])
    editor = EasyEditEditor.from_hparams(hparams)
    exp = EditExperimentRunner(hparams, editor.batch_edit)
    LOG.info(f"Use {len(edit_payload)} prompts to be edited")
    metrics, edited_model = exp.edit_and_eval(
        edit_payload,
        test_data,
        skip_eval=True,
    )
    wandb.log(metrics)


def run():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    llama_guard_edit()


if __name__ == "__main__":
    run()
