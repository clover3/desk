import fire
import logging
from desk_util.clf_util import clf_predict_w_predict_fn
from desk_util.runnable.run_eval_clf import run_eval_clf
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.display.show_avg_p_r_f1 import show_avg_p_r_f1
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_2024_split_subreddit_list, get_group2_list
from taskman_client.wrapper3 import JobContext

LOG = logging.getLogger(__name__)


def predict_sb_split(
        run_name_fmt: str,
        do_eval=False,
        overwrite=False,
        dataset_fmt = "{}_3_val_1000"

) -> None:
    job_name = run_name_fmt + "_Group2"
    with JobContext(job_name):
        todo = get_group2_list()
        for sb in todo:
            try:
                run_name = run_name_fmt.format(sb)
                dataset = dataset_fmt.format(sb)
                predict_fn = get_classifier(run_name)
                clf_predict_w_predict_fn(dataset, run_name, predict_fn, overwrite)
                if do_eval:
                    run_eval_clf(run_name, dataset,
                                 True)
            except FileNotFoundError as e:
                print(e)



if __name__ == "__main__":
    fire.Fire(predict_sb_split)
