import logging

import fire
from sklearn.metrics import accuracy_score

from chair.misc_lib import average
from desk_util.path_helper import load_clf_pred
from rule_gen.reddit.path_helper import get_split_subreddit_list, load_2024_split_subreddit_list

LOG = logging.getLogger(__name__)


def predict_sb_split(
        run_name_fmt1: str,
        run_name_fmt2: str,
        split: str,
        dataset_fmt="{}_3_val_1000"

) -> None:
    if "2024" in dataset_fmt:
        todo = load_2024_split_subreddit_list(split)
    else:
        todo = get_split_subreddit_list(split)

    acc_list = []
    for sb in todo:
        try:
            run_name1 = run_name_fmt1.format(sb)
            run_name2 = run_name_fmt2.format(sb)

            dataset = dataset_fmt.format(sb)
            preds1 = load_clf_pred(dataset, run_name1)
            preds1 = [e[1] for e in preds1]
            preds2 = load_clf_pred(dataset, run_name2)
            preds2 = [e[1] for e in preds2]
            a1 = accuracy_score(preds1, preds2)
            a2 = accuracy_score(preds2, preds1)
            assert abs(a1 - a2) < 0.01
            print(sb, a1)
            acc_list.append(a1)
        except FileNotFoundError as e:
            print(e)
        except ValueError as e:
            print(sb)
            print(e)
            # raise

    # bert2 vs chatgpt_{}_none : 0.6
    # bert2 vs random: 0.5
    # chatgpt_none vs chatgpt_{}_none: 0.67
    print("Avg similarity", average(acc_list))


if __name__ == "__main__":
    fire.Fire(predict_sb_split)
