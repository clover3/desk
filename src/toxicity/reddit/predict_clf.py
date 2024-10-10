import torch
import fire

from toxicity.clf_util import clf_predict_w_predict_fn, clf_predict_w_batch_predict_fn
from toxicity.path_helper import get_model_save_path
from toxicity.runnable.run_eval_clf import run_eval_clf
from transformers import pipeline

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_classifier(run_name):
    model_path = get_model_save_path(run_name)
    pipe = pipeline("text-classification", model=model_path, device=get_device())

    label_map = {
        "LABEL_0": "0",
        "LABEL_1": "1",
    }

    def predict(text):
        r = pipe(text, truncation=True)[0]
        label = label_map[r["label"]]
        score = r["score"]
        if label == "0":
            score = -score
        return label, score

    def batch_predict(text_list):
        for r in pipe(text_list, truncation=True):
            label = label_map[r["label"]]
            score = r["score"]
            if label == "0":
                score = -score
            yield label, score

    return predict


def predict_clf_main(
        run_name: str,
        dataset: str,
        do_eval=False,
        do_report=False,
        print_metrics=""
) -> None:
    predict_fn = get_classifier(run_name)
    clf_predict_w_predict_fn(dataset, run_name, predict_fn)
    if do_eval:
        run_eval_clf(run_name, dataset,
                     do_report, print_metrics)


if __name__ == "__main__":
    fire.Fire(predict_clf_main)
