import itertools
import os

from tqdm import tqdm

from toxicity.cpath import output_root_path
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import save_csv, read_csv
from toxicity.llama_guard.load_llama_guard import load_llg2, load_llg2_custom
from toxicity.path_helper import get_dataset_pred_save_path
from toxicity.predictors.api_based import load_api_1, load_api_based_predictor_by_name
from toxicity.predictors.instruct_as_guard import load_instruct_as_guard
from toxicity.predictors.template_debug import load_ilu


def run_toxigen_prediction(predict_fn, run_name, split="train", n_pred=None):
    if n_pred is None:
        dataset_name: str = f'toxigen_{split}'
        test_dataset: ToxigenBinary = ToxigenBinary(split)
        payload = list(test_dataset)
    else:
        dataset_name: str = f'toxigen_{split}_head_{n_pred}'
        test_dataset: ToxigenBinary = ToxigenBinary(split)
        payload = itertools.islice(test_dataset, n_pred)
        payload: list[dict] = list(payload)

    def predict(e):
        text, score = predict_fn([e['text']])
        return e['id'], text, score

    pred_itr = map(predict, tqdm(payload, desc="Processing", unit="item"))
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    save_csv(pred_itr, save_path)
    print(f"Saved at {save_path}")


def get_llama_guard_like_predictor(run_name):
    if run_name == "llama_guard2_prompt":
        return load_llg2()
    elif run_name.startswith("lg2_ft_"):
        n = int(run_name[len("lg2_ft_"):])
        model_path = f"/home/qdb5sn/work/LLaMA-Factory/saves/lg2/lora/lg_toxigen_{n}/"
        return load_llg2(model_path)
    elif run_name.startswith("lgt2_"):
        n = int(run_name[len("lgt2_"):])
        model_path = f"/home/qdb5sn/work/LLaMA-Factory/saves/lg2/lora/lgt2_{n}/"
        return load_llg2(model_path)
    elif run_name.startswith("lgt3_"):
        model_path = f"/home/qdb5sn/work/LLaMA-Factory/saves/lg2/lora/{run_name}/"
        return load_llg2(model_path)
    elif run_name.startswith("ft"):
        model_path = f"/home/qdb5sn/work/LLaMA-Factory/saves/{run_name}/"
        return load_llg2(model_path)

    elif run_name.startswith("lg2_"):
        group_str = ", ".join(['latino', 'asian', 'chinese', 'lgbtq', 'women', 'mental disabled', 'physical disabled', 'jewish', 'middle_east', 'muslim', 'mexican', 'black', 'native_american'])
        desc_d = {
            "lg2_1": "A rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion.",
            "lg2_2": "Toxicity is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
                     "Imagine that this is a part of a discussion. Consider if this is hateful against minority groups.",
            "lg2_3": "Toxicity is a rude, disrespectful, or unreasonable comment that is likely to make people leave a discussion."
                     f"Imagine that this is a part of a discussion. Consider if this is hateful against minority groups: {group_str}",
        }
        return load_llg2_custom(desc_d[run_name])
    elif run_name == "llama3_instruct6":
        return load_instruct_as_guard()
    elif run_name.startswith("instruct_"):
        sub_name = run_name[len("instruct_"):]
        return load_instruct_as_guard(model_id=sub_name)
    elif run_name.startswith("ilu"):
        return load_ilu(run_name)
    elif run_name.startswith("api_"):
        return load_api_based_predictor_by_name(run_name)
    else:
        raise ValueError(f"Unknown run_name: {run_name}")
