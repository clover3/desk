import fire

from toxicity.clf_util import clf_predict_w_predict_fn
from toxicity.reddit.corpus_sim.compute_sim import get_most_sim
from toxicity.reddit.path_helper import get_split_subreddit_list
from toxicity.reddit.predict_clf import get_classifier
from toxicity.runnable.run_eval_clf import run_eval_clf


def predict_clf_main(
        trg_reddit: str,
        do_eval=True,
        do_report=True,
        print_metrics=""
) -> None:
    dataset = f"{trg_reddit}_val"
    avail = get_split_subreddit_list("train")
    most_sim = get_most_sim(trg_reddit, avail)
    run_name = f"bert_{most_sim}"
    print(f"{most_sim} is most similar for {trg_reddit}")
    predict_fn = get_classifier(run_name)
    report_name = "sim_" + run_name
    clf_predict_w_predict_fn(dataset, report_name, predict_fn)
    if do_eval:
        run_eval_clf(report_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
