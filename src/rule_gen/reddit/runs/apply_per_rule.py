import os

from desk_util.clf_util import clf_predict_w_predict_fn
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import load_subreddit_list, get_n_rules


def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        try:
            n_rule = get_n_rules(sb)
            for rule_idx in range(n_rule):
                run_name = f"chatgpt_sr_{sb}_{rule_idx}_both"
                dataset = f"{sb}_val_100"
                save_path = get_clf_pred_save_path(run_name, dataset)
                if os.path.exists(save_path):
                    print("Skip", run_name)
                    continue

                predict_fn = get_classifier(run_name)
                clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()