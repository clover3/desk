from chair.list_lib import left
from toxicity.llama_guard.output_convertor import parse_predictions
from toxicity.dataset_helper.load_toxigen import ToxigenBinary
from toxicity.io_helper import read_csv, save_csv
from toxicity.path_helper import get_dataset_pred_save_path, get_comparison_save_path
from toxicity.toxigen.toxigen_eval_analysis.run_eval import get_dataset_split



def load_pred_labels(n_pred, split, run_name):
    ds_split = get_dataset_split(split)
    if n_pred is None:
        dataset_name: str = f'toxigen_{split}'
    else:
        dataset_name: str = f'toxigen_{split}_head_{n_pred}'

    print(f"dataset_name={dataset_name}")
    print(f"split={split}")
    save_path: str = get_dataset_pred_save_path(run_name, dataset_name)
    preds = read_csv(save_path)
    text_predictions = [e[1] for e in preds]
    raw_scores = [float(e[2]) for e in preds]
    # Convert to binary predictions
    target_string = "S1"
    parsed: list[tuple[int, float]] = parse_predictions(text_predictions, raw_scores, target_string)
    predictions = left(parsed)
    n_item = len(parsed)
    test_dataset: ToxigenBinary = ToxigenBinary(ds_split)
    return dataset_name, n_item, predictions, test_dataset


def run_toxigen_eval(run_name, base_split, para_split, n_pred=None):
    dataset_name1, _n_item1, preds1, ds1 = load_pred_labels(n_pred, base_split, run_name)
    dataset_name2, _n_item2, preds2, ds2 = load_pred_labels(None, para_split, run_name)

    text_d = {}
    label_d = {}
    pred_d = {}
    for p, ds in zip(preds1, ds1):
        data_id = ds['id']
        text_d[data_id] = ds['text']
        label_d[data_id] = ds['label']
        pred_d[data_id] = p

    rows = []
    for i in range(_n_item2):
        data_id = ds2[i]['id']
        print(data_id)
        label2 = ds2[i]['label']
        label1 = label_d[data_id]
        pred1 = pred_d[data_id]
        assert label1 == label2

        row = [label1, pred1, text_d[data_id], preds2[i], ds2[i]['text']]
        rows.append(row)

    save_path = get_comparison_save_path(run_name, dataset_name2)
    save_csv(rows, save_path)


def main():
    run_toxigen_eval("lgt2_1000", "train", "1000_para", 1000)


if __name__ == "__main__":
    main()