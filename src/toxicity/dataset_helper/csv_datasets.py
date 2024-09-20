import os
from toxicity.cpath import output_root_path
from toxicity.io_helper import read_csv
from toxicity.path_helper import get_label_path


def load_toxigen_para():
    dataset_name = "toxigen_head_100_para_clean"
    data = load_toxigen_like_csv(dataset_name)
    return data


def load_toxigen_like_csv(dataset_name):
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset_name}.csv")
    id_text_list = read_csv(save_path)
    rows = read_csv(get_label_path(dataset_name))
    labels_d = {data_id: int(label) for data_id, label in rows}
    data = [{"text": text, "label": labels_d[data_id]} for data_id, text in id_text_list]
    return data
