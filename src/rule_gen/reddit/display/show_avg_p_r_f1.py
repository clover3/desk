import fire
from chair.tab_print import print_table
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.reddit.corpus_sim.compute_sim import get_most_sim
from rule_gen.reddit.path_helper import get_split_subreddit_list
from desk_util.runnable.run_eval import load_labels, clf_eval


def calculate_stats(numbers):
    n = len(numbers)
    if n == 0:
        return None, None
    mean = sum(numbers) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in numbers)
    variance = squared_diff_sum / n
    stdev = variance ** 0.5
    return mean, stdev


def show_avg_p_r_f1(dataset_fmt, get_run_name, split):
    columns = ["f1", "precision", "recall"]
    n_expected = len(get_split_subreddit_list(split))
    score_l_d = compute_per_reddit_scores(columns, dataset_fmt, get_run_name, split)

    row = []
    head = []
    for c in columns:
        scores = ", ".join(f"{s:.2f}" for s in score_l_d[c])
        print(f"{c}: [{scores}]")
        n_item = len(score_l_d[c])
        if not n_item == n_expected:
            raise ValueError("Expected {} but got {}".format(n_expected, n_item))
        avg, std = calculate_stats(score_l_d[c])
        avg = "{0:.2f}".format(avg)
        std = "{0:.2f}".format(std)
        head.extend([c + "_avg", c + "_std"])
        row.extend([avg, std])
    print_table([row])


def compute_per_reddit_scores(columns, dataset_fmt, get_run_name, split):
    subreddit_list = get_split_subreddit_list(split)
    score_l_d = {c: list() for c in columns}
    for sb in subreddit_list:
        run_name = get_run_name(sb)
        dataset = dataset_fmt.format(sb)
        try:
            save_path: str = get_clf_pred_save_path(run_name, dataset)
            raw_preds = read_csv(save_path)
            preds = [(data_id, int(pred), float(score)) for data_id, pred, score in raw_preds]
            labels = load_labels(dataset)
            score_d = clf_eval(preds, labels)
            for c in columns:
                score_l_d[c].append(score_d[c])
        except FileNotFoundError as e:
            print(e)
            raise
        except ValueError:
            print(save_path)
            raise
    return score_l_d


def display_result(get_run_name):
    dataset_fmt = "{}_val_100"
    split = "val"
    show_avg_p_r_f1(dataset_fmt, get_run_name, split)


def display_from_run_name_fmt(run_name_fmt):
    print("run_name_fmt: {}".format(run_name_fmt))
    def get_run_name(dataset):
        return run_name_fmt.format(dataset)

    display_result(get_run_name)


def most_sim_run():
    print("most_sim_run")
    avail = get_split_subreddit_list("train")

    def get_run_name(target_sb):
        most_sim = get_most_sim(target_sb, avail)
        run_name = f"sim_bert_{most_sim}"
        return run_name

    display_result(get_run_name)


def bert_train_mix():
    print("bert_train_mix")

    def get_run_name(target_sb):
        return "bert_train_mix3"

    display_result(get_run_name)


def main(run_name_fmt, dataset_fmt="{}_val_100", split="val"):
    show_avg_p_r_f1(dataset_fmt, run_name_fmt.format, split)


if __name__ == "__main__":
    fire.Fire(main)
