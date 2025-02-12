from desk_util.clf_util import clf_predict_w_predict_fn
from rule_gen.reddit.path_helper import get_n_rules
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier


def main(sb="churning"):
    n_rule = get_n_rules(sb)
    for rule_idx in range(n_rule):
        run_name = f"api_srr_{sb}_{rule_idx}"
        print(run_name)
        predict_fn = get_classifier(run_name)
        dataset = f"{sb}_val_100"
        clf_predict_w_predict_fn(dataset, run_name, predict_fn)


if __name__ == "__main__":
    main()
