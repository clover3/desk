import fire
from chair.tab_print import print_table
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_clf_pred_save_path
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.runnable.run_eval import load_labels, clf_eval


def print_perf(dataset_fmt, run_name_fmt, split):
    subreddit_list = get_split_subreddit_list(split)
    output = []
    columns = ["f1", "precision", "recall"]
    head = [""] + columns
    output.append(head)
    for sb in subreddit_list:
        run_name = run_name_fmt.format(sb)
        dataset = dataset_fmt.format(sb)
        row = [sb]
        try:
            save_path: str = get_clf_pred_save_path(run_name, dataset)
            raw_preds = read_csv(save_path)
            preds = [(data_id, int(pred), float(score)) for data_id, pred, score in raw_preds]
            labels = load_labels(dataset)
            score_d = clf_eval(preds, labels)
            for t in columns:
                row.append("{0:.2f}".format(score_d[t]))
        except FileNotFoundError:
            row += ["-"] * len(columns)
        except ValueError:
            print(save_path)
            raise
        output.append(row)
    print_table(output)


def main(run_name_fmt, split = "val", dataset_fmt = "{}_val_100"):
    # run_name_fmt = "bert_train_mix3"
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)
    print_perf(dataset_fmt, run_name_fmt, split)


if __name__ == "__main__":
    fire.Fire(main)
