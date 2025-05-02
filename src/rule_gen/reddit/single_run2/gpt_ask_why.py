import json

import fire
from tqdm import tqdm

from chair.misc_lib import make_parent_exists
from desk_util.clf_util import load_csv_dataset_by_name
from desk_util.io_helper import save_csv
from desk_util.path_helper import get_clf_pred_save_path
from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.questions.llm_inf_saver import get_llm_predictor


def main(sb="fantasyfootball"):
    run_name = "chatgpt_why_{}".format(sb)
    dataset = "{}_2_val_100".format(sb)
    payload = load_csv_dataset_by_name(dataset)
    predict_fn = get_llm_predictor(run_name)
    clf_preds = []
    output = []
    for data_id, text in tqdm(payload):
        save_e = predict_fn(text)
        label, score, ret_text = save_e
        clf_preds.append((data_id, label, score))
        output.append(save_e)

    save_path = get_clf_pred_save_path(run_name, dataset)
    save_csv(clf_preds, save_path)

    res_save_path = os.path.join(
        output_root_path, "reddit",
        "j_res", dataset, f"{run_name}.json")
    make_parent_exists(res_save_path)
    json.dump(output, open(res_save_path, "w"), indent=4)


if __name__ == "__main__":
    fire.Fire(main)
