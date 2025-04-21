import json
import os
from typing import Callable, Any

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
import os 

from tqdm import tqdm

from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import save_csv, read_csv
from desk_util.path_helper import get_clf_pred_save_path


def llm_predict_w_predict_fn(
        dataset, run_name,
        predict_fn: Callable[[str], tuple[int, Any, str]],
        overwrite_existing=False,
):
    payload = load_csv_dataset_by_name(dataset)
    save_path = get_clf_pred_save_path(run_name, dataset)

    if not overwrite_existing and os.path.exists(save_path):
        if len(read_csv(save_path)) == len(payload):
            print(f"Prediction exists. Skip prediction")
            print(f": {save_path}")
            return
        else:
            print(f"Prediction exists but not complete. Overwritting")
            print(f": {save_path}")
            
    clf_preds = []
    output = []
    for data_id, text in tqdm(payload):
        label, score, ret_text = predict_fn(text)
        clf_preds.append((data_id, label, score))
        save_e = data_id, label, score, ret_text
        output.append(save_e)
        
    save_csv(clf_preds, save_path)
    print(f"Clf pred saved at {save_path}")
    res_save_path = os.path.join(
        output_root_path, "reddit",
        "gpt_res", dataset, f"{run_name}.json")
    make_parent_exists(res_save_path)
    json.dump(output, open(res_save_path, "w"), indent=4)


