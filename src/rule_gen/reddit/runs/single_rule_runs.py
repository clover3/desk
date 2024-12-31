from desk_util.clf_util import clf_predict_w_predict_fn
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from desk_util.runnable.run_eval_clf import run_eval_clf


def apply_single_rule_to_train_split(rule_idx, rule_sb, do_eval=False, do_report=False):
    run_name = f"api_sr_{rule_sb}_{rule_idx}_detail"
    predict_fn = get_classifier(run_name)
    sb_split = "train"
    print(run_name)
    apply_classifier_to_sb_splits(predict_fn, run_name, sb_split, do_eval, do_report)


def apply_classifier_to_sb_splits(predict_fn, run_name, sb_split, do_eval, do_report):
    subreddit_list = get_split_subreddit_list(sb_split)
    for sb in subreddit_list:
        dataset = f"{sb}_val_100"
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        if do_eval:
            run_eval_clf(run_name, dataset,
                         do_report, )


def main():
    rule_sb = "pics"
    rule_idx = 7
    apply_single_rule_to_train_split(rule_idx, rule_sb)


if __name__ == "__main__":
    main()

