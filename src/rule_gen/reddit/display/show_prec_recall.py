import fire
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.path_helper import get_split_subreddit_list
from desk_util.runnable.run_eval import load_labels, clf_eval


def print_perf(dataset_fmt, run_name_fmt, split, columns):
    if split.endswith("_30"):
        split = split[:-3]
        subreddit_list = get_split_subreddit_list(split)[:30]
    else:
        subreddit_list = get_split_subreddit_list(split)
    output = []
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
    columns = ["f1", "precision", "recall"]
    print_perf(dataset_fmt, run_name_fmt, split, columns)


if __name__ == "__main__":
    fire.Fire(main)
