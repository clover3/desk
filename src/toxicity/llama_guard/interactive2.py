import fire

from toxicity.llama_guard.load_llama_guard import load_llg2
from toxicity.predictors.get_predictor import get_llama_guard_like_predictor


def main(
) -> None:
    run_name = "ft_trump"
    predict_fn = get_llama_guard_like_predictor(run_name)
    while True:
        prompt = input("Enter a prompt: ").strip(" \r\t\n")
        result, score = predict_fn([prompt])
        print(result, score)


if __name__ == "__main__":
    fire.Fire(main)
