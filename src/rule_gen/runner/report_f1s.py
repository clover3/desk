import fire

from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from desk_util.runnable.run_eval import load_labels, clf_eval
from rule_gen.reddit.path_helper import get_split_subreddit_list
from taskman_client.task_proxy import get_task_manager_proxy


def main(run_name_fmt, dataset_fmt="{}_2_val_100", metric = "f1", split="all"):
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)
    proxy = get_task_manager_proxy()
    subreddit_list = get_split_subreddit_list(split)
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
            print(sb, score_d[metric], )
            proxy.report_number(run_name, score_d[metric], dataset, metric)

        except FileNotFoundError as e :
            print(e)
            pass
        except ValueError:
            print(save_path)


if __name__ == "__main__":
    fire.Fire(main)
