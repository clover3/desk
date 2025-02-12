import fire

from taskman_client.task_proxy import get_task_manager_proxy
from rule_gen.reddit.display.show_avg_p_r_f1 import compute_per_reddit_scores, calculate_stats


def main(run_name_fmt):
    # run_name_fmt = "api_{}_none"
    dataset_fmt = "{}_val_100"
    field = "f1"
    columns = [field]
    split = "val"
    n_expected = 20
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)

    def get_run_name(dataset):
        return run_name_fmt.format(dataset)

    score_l_d = compute_per_reddit_scores(columns, dataset_fmt, get_run_name, split)
    n_item = len(score_l_d[field])
    assert n_item == n_expected
    avg, std = calculate_stats(score_l_d[field])
    print("Avg\t{}".format(avg))
    proxy = get_task_manager_proxy()
    dataset = "val_" + dataset_fmt
    proxy.report_number(run_name_fmt, avg, dataset, "f1-avg")


if __name__ == "__main__":
    fire.Fire(main)

