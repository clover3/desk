import logging
from easyeditor.editors.edit_runner import EasyEditEditor
from taskman_client.task_proxy import get_task_manager_proxy
from toxicity.ee.easy_edit_main import get_hparams
from toxicity.ee.edit_exp_runner import EditExperimentRunner
from toxicity.dataset_helper.load_toxigen import load_toxigen_formatted
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
    edit_payload = load_toxigen_formatted(n_item=100)
    test_data = load_toxigen_formatted(split="test", n_item=100)
    editor = EasyEditEditor.from_hparams(hparams)
    exp = EditExperimentRunner(hparams, editor.batch_edit)
    LOG.info(f"{len(edit_payload)} prompts")
    metrics, edited_model = exp.run_edit_and_eval2(
        edit_payload,
        test_data,
        do_post_eval=True,
    )
    print(metrics)
    run_name = f"ee_ft_l{target_layer}"
    proxy = get_task_manager_proxy()
    proxy.report_number(run_name, float(metrics["post"]["train_acc"]), "toxigen_train_head_100", "acc")
    proxy.report_number(run_name, float(metrics["post"]["valid_acc"]), "toxigen_test_head_100", "acc")




if __name__ == "__main__":
    fire.Fire(main)
