from desk_util.clf_util import clf_predict_w_predict_fn
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.classifier_loader.load_by_name import get_classifier



def apply_gnq2_to_sb_splits(run_name_itr, dataset_fmt, sb_split):
    subreddit_list = get_split_subreddit_list(sb_split)
    for sb in subreddit_list:
        print(sb)
        dataset = dataset_fmt.format(sb)
        for run_name in run_name_itr:
            print(run_name)
            predict_fn = get_classifier(run_name)
            clf_predict_w_predict_fn(dataset, run_name, predict_fn)


def main():
    run_name_itr = [f"chatgpt_v2_gnq2_{rule_idx}" for rule_idx in range(18)]
    dataset_fmt = "{}_2_val_100"
    apply_gnq2_to_sb_splits(run_name_itr, dataset_fmt, sb_split="train")


if __name__ == "__main__":
    main()

