import fire
from sklearn.metrics import f1_score, confusion_matrix

from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from desk_util.runnable.run_eval import load_labels, align_preds_and_labels
from rule_gen.reddit.path_helper import get_split_subreddit_list
from rule_gen.reddit.s9.clf_for_one import compute_f_k


def main(run_name_fmt, dataset_fmt="{}_2_val_100"):
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)
    for split in ["train", "val"]:
        subreddit_list = get_split_subreddit_list(split)
        table = []
        for sb in subreddit_list:
            run_name = run_name_fmt.format(sb)
            dataset = dataset_fmt.format(sb)
            try:
                save_path: str = get_clf_pred_save_path(run_name, dataset)
                raw_preds = read_csv(save_path)
                preds = [(data_id, int(pred), float(score)) for data_id, pred, score in raw_preds]
                labels = load_labels(dataset)
                aligned_preds, aligned_labels, aligned_scores = align_preds_and_labels(preds, labels)

                f1 = f1_score(aligned_labels, aligned_preds)
                f19 = compute_f_k(aligned_labels, aligned_preds, 19)
                row = [sb, f1, f19]
                table.append(row)
            except FileNotFoundError:
                pass
            except ValueError:
                print(save_path)
        print_table(table)


if __name__ == "__main__":
    fire.Fire(main)
