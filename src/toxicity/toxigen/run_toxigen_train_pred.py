import tqdm
from tqdm import tqdm
import itertools
from typing import Callable
from toxicity.llama_guard.load_llama_guard import load_llg2
from toxicity.path_helper import get_dataset_pred_save_path
from toxicity.dataset_helper.load_toxigen import ToxigenTrain
from toxicity.io_helper import save_csv


def predidct_toxigen_train_head(predict_fn, run_name, n_pred=900):
    dataset_name: str = f'toxigen_tt_head_{n_pred}'
    test_dataset: ToxigenTrain = ToxigenTrain()
    payload = itertools.islice(test_dataset, n_pred)
    payload = list(payload)

    def predict(e):
        text, score = predict_fn([e['text']])
        return e['id'], text, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def main_llama_guard2_prompt() -> None:
    run_name: str = "llama_guard2_prompt"
    llg: Callable[[list[str]], tuple[str, float]] = load_llg2()
    predidct_toxigen_train_head(llg, run_name, 100)



if __name__ == "__main__":
    main_llama_guard2_prompt()
