import numpy as np

from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_clf_pred_save_path

from collections import Counter

import fire

from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.path_helper import get_split_subreddit_list



def convert_entry(entry):
    data_id, pred, score_s = entry
    prob = np.exp(float(score_s))
    if int(pred):
        score = prob
    else:
        # Higher score -> Lower confidence
        score = -prob

    return data_id, pred, score


def convert_for_dataset(run_name, dataset):
    try:
        save_path: str = get_clf_pred_save_path(run_name, dataset)
        raw_preds = read_csv(save_path)

        score_s_list = [e[2] for e in raw_preds]
        has_score = not all([s == "0" for s in score_s_list])
        if has_score:
            new_run_name = "c_" + run_name
            items = [convert_entry(e) for e in raw_preds]
            save_path: str = get_clf_pred_save_path(new_run_name, dataset)
            save_csv(items, save_path)
            print("saved at ", save_path)
    except FileNotFoundError:
        pass
    except ValueError:
        pass


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
                for sb in subreddit_list:
                    run_name = run_name_fmt.format(sb)
                    dataset = dataset_fmt.format(sb)
                    convert_for_dataset(run_name, dataset)


if __name__ == "__main__":
    fire.Fire(main)
