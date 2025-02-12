
from desk_util.clf_util import clf_predict_w_predict_fn
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_n_rules


def main():
    subreddit_list = get_split_subreddit_list("train")
    start = subreddit_list.index("Incels")
    todo = subreddit_list[start+1:]

    for sb in todo:
        try:
            n_rule = get_n_rules(sb)

            for rule_idx in range(n_rule):
                run_name = f"api_sr_{sb}_{rule_idx}_both"
                predict_fn = get_classifier(run_name)
                dataset = f"{sb}_val_100"
                clf_predict_w_predict_fn(dataset, run_name, predict_fn)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
