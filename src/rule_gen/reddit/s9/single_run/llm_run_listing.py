import logging
import os

from chair.tab_print import print_table
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_j_res_save_path

LOG = logging.getLogger(__name__)


def check_for(
        run_name_fmt: str,
        get_path_fn,
) -> None:
    table = []
    for dataset_fmt in ["{}_2_train_100", "{}_2_val_100"]:
        for split in ["train", "val"]:
            todo = get_split_subreddit_list(split)
            n_exist = 0
            for sb in todo:
                run_name = run_name_fmt.format(sb)
                dataset = dataset_fmt.format(sb)
                p = get_path_fn(run_name, dataset)

                if os.path.exists(p):
                    n_exist += 1

            if n_exist == len(todo):
                state = "Done"
            elif n_exist > 0:
                state = "{}/{}".format(n_exist, len(todo))
            else:
                state = "-"

            row = [run_name_fmt, split, dataset_fmt, state]
            table.append(row)
    print_table(table)


def main():
    print(" J_Res")
    check_for("llg_toxic", get_j_res_save_path)
    check_for("llg_default", get_j_res_save_path)
    check_for("llama_s9nosb", get_j_res_save_path)
    # print(" CLF pred")
    # check_for("llg_toxic", get_clf_pred_save_path)
    # check_for("llg_default", get_clf_pred_save_path)


if __name__ == "__main__":
    main()
