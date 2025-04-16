from collections import Counter

import fire

from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    run_name_fmt_list = [
        "chatgpt_none",
        "chatgpt_{}_none",
        "chatgpt_{}_both",
    ]
    dataset_fmt_list = ["{}_3_val_1000", "{}_3_test_1000"]
    for run_name_fmt in run_name_fmt_list:
        for dataset_fmt in dataset_fmt_list:
            print("run_name_fmt", run_name_fmt)
            print("dataset_fmt", dataset_fmt)
            for split in ["train", "val"]:
                subreddit_list = get_split_subreddit_list(split)

                counter = Counter()
                for sb in subreddit_list:
                    run_name = run_name_fmt.format(sb)
                    dataset = dataset_fmt.format(sb)
                    row = [sb]
                    try:
                        save_path: str = get_clf_pred_save_path(run_name, dataset)
                        raw_preds = read_csv(save_path)

                        score_s_list = [e[2] for e in raw_preds]
                        has_score = not all([s == "0" for s in score_s_list])
                        if has_score:
                            counter["has_score"] += 1
                        else:
                            counter["no_score"] += 1
                    except FileNotFoundError:
                        pass
                    except ValueError:
                        print(save_path)
                print(split, counter)


if __name__ == "__main__":
    fire.Fire(main)
