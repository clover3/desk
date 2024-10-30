import fire

from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.reddit.classifier_loader.load_by_name import get_classifier
from toxicity.runnable.run_eval_clf import run_eval_clf


def apply_classifier_to_sb_splits(predict_fn, run_name, sb_split, do_eval=True, do_report=False):
    subreddit_list = get_split_subreddit_list(sb_split)
    for sb in subreddit_list:
        dataset = f"{sb}_val_100"
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         do_report, )


def main(run_name):
    predict_fn = get_classifier(run_name)
    apply_classifier_to_sb_splits(predict_fn, run_name, "train")


if __name__ == "__main__":
    fire.Fire(main)

