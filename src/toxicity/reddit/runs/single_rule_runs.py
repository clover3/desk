from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.reddit.classifier_loader.load_by_name import get_classifier


def apply_single_rule_to_train_split(rule_idx, rule_sb):
    subreddit_list = get_split_subreddit_list("train")
    run_name = f"api_sr_{rule_sb}_{rule_idx}_detail"
    print(run_name)
    predict_fn = get_classifier(run_name)
    for sb in subreddit_list:
        dataset = f"{sb}_val_100"
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        # if do_eval:
        #     run_eval_clf(run_name, dataset,
        #                  do_report, print_metrics)
        #
        #
        break

def main():
    rule_sb = "pics"
    rule_idx = 7
    apply_single_rule_to_train_split(rule_idx, rule_sb)


if __name__ == "__main__":
    main()

