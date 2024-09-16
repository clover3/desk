import logging
import os
import sys

from easyeditor import ALG_HP_DICT
from easyeditor.editors.edit_runner import EasyEditEditor
from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.ee.edit_exp_runner import EditExperimentRunner
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted, load_toxigen_para_formatted

LOG = logging.getLogger(__name__)


def get_hparams(hparams_name_or_path: str):
    dir_name = os.path.dirname(hparams_name_or_path)
    algo_name = os.path.basename(dir_name)
    LOG.info("algo_name=%s", algo_name)
    params_class = ALG_HP_DICT[algo_name]
    return params_class.from_hparams(hparams_name_or_path)


def edit_exp(hparams, run_name=""):
    edit_payload = load_toxigen_formatted(n_item=100)
    test_data = load_toxigen_formatted(split="test", n_item=100)
    para_data = load_toxigen_para_formatted()
    eval_data_d = {
        "test": test_data,
        "para": para_data
    }
    editor = EasyEditEditor.from_hparams(hparams)
    exp = EditExperimentRunner(hparams, editor.batch_edit)
    LOG.info(f"{len(edit_payload)} prompts")
    metrics, edited_model = exp.run_edit_and_eval2(
        edit_payload,
        eval_data_d=eval_data_d,
        do_pre_eval=False,
        do_post_eval=True,
        edit_only_fail=hparams.edit_only_fail,
    )
    print(metrics)
    post_metrics = metrics["post"]
    if run_name:
        proxy = get_task_manager_proxy()
        todo = {
            "train_acc": "toxigen_train_head_100",
            "test_acc": "toxigen_test_head_100",
            "para_acc": "toxigen_head_100_para_clean",
        }
        for key, long_name in todo.items():
            proxy.report_number(run_name, float(post_metrics[key]), long_name, "acc")


def run():
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    LOG.info(__name__)
    try:
        hp_path = sys.argv[1]
        run_name = sys.argv[2]
    except IndexError:
        hp_path = 'confs/EasyEdit/hparams/LoRA/llama_guard2.yaml'
        run_name = ""


    hparams = get_hparams(hp_path)
    LOG.info("%s", str(hparams))
    edit_exp(hparams, run_name)


if __name__ == "__main__":
    run()
