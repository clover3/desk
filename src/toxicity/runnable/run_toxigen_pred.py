import fire
from toxicity.predictors.get_predictor import run_toxigen_prediction, get_llama_guard_like_predictor
from toxicity.toxigen_eval_analysis.run_eval import run_toxigen_eval


def main(
        run_name: str,
        split: str,
        n_pred=None,
        do_eval=False,
) -> None:
    predict_fn = get_llama_guard_like_predictor(run_name)
    run_toxigen_prediction(predict_fn, run_name, split, n_pred)
    if do_eval:
        run_toxigen_eval(run_name, split, do_report=True, n_pred=n_pred)


if __name__ == "__main__":
    fire.Fire(main)
