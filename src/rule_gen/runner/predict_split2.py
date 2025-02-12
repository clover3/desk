import fire
import logging
from desk_util.clf_util import clf_predict_w_predict_fn
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.display.show_avg_p_r_f1 import show_avg_p_r_f1
from rule_gen.reddit.path_helper import get_split_subreddit_list

LOG = logging.getLogger(__name__)


def predict_sb_split(
        run_name_fmt: str,
        split: str,
) -> None:
    todo = get_split_subreddit_list(split)
    dataset_fmt = "{}_2_val_100"

    for sb in todo:
        run_name = run_name_fmt.format(sb)
        dataset = dataset_fmt.format(sb)
        predict_fn = get_classifier(run_name)
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)

    show_avg_p_r_f1(dataset_fmt, run_name_fmt.format, split)


if __name__ == "__main__":
    fire.Fire(predict_sb_split)
