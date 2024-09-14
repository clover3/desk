import logging
import os
import sys

from easyeditor import ALG_HP_DICT
from easyeditor.editors.edit_runner import EasyEditEditor
from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.ee.edit_exp_runner import EditExperimentRunner
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted
LOG = logging.getLogger(__name__)


def get_hparams(hparams_name_or_path: str):
    dir_name = os.path.dirname(hparams_name_or_path)
    algo_name = os.path.basename(dir_name)
    LOG.info("algo_name=%s", algo_name)
    params_class = ALG_HP_DICT[algo_name]
    return params_class.from_hparams(hparams_name_or_path)


def llama_guard_edit():
    LOG.info(__name__)
    if len(sys.argv) > 1:
        hp_path = sys.argv[1]
    else:
        hp_path = 'confs/EasyEdit/hparams/LoRA/llama_guard2.yaml'
    hparams = get_hparams(hp_path)
    LOG.info("%s", str(hparams))
    edit_payload = load_toxigen_formatted(n_item=100)
    test_data = load_toxigen_formatted(split="test", n_item=100)
    editor = EasyEditEditor.from_hparams(hparams)
    exp = EditExperimentRunner(hparams, editor.batch_edit)
    LOG.info(f"{len(edit_payload)} prompts")
    metrics, edited_model = exp.edit_and_eval(
        edit_payload,
        test_data,
        # skip_eval=True,
    )
    print(metrics)

    post_metrics = metrics["post"]
    do_report = "accuracy" in post_metrics
    if do_report:
        run_name = ""
        proxy = get_task_manager_proxy()
        acc = metrics["acc"]
        proxy.report_number(run_name, acc, "toxigen_train_head_100", "acc")


def run():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    llama_guard_edit()


if __name__ == "__main__":
    run()
