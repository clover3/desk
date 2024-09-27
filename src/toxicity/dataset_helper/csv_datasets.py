import os
from datasets import Dataset

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


def load_csv_as_hf_dataset(dataset_name: str) -> Dataset:
    save_path: str = os.path.join(output_root_path, "datasets", f"{dataset_name}.csv")
    id_text_list = read_csv(save_path)
    rows = read_csv(get_label_path(dataset_name))
    labels_d = {data_id: int(label) for data_id, label in rows}

    payload = []
    for row in id_text_list:
        data_id, text = row
        e = {
            "id": data_id,
            "label": labels_d[data_id],
            "text": text,
        }
        payload.append(e)

    return Dataset.from_list(payload)

#
# class CSVDataset(Dataset):
#     def __init__(self, dataset_name):
#         save_path: str = os.path.join(output_root_path, "datasets", f"{dataset_name}.csv")
#         id_text_list = read_csv(save_path)
#         rows = read_csv(get_label_path(dataset_name))
#         labels_d = {data_id: int(label) for data_id, label in rows}
#         payload = []
#         for row in id_text_list:
#             data_id, text = row
#             e = {
#                 "id": data_id,
#                 "label": labels_d[data_id],
#                 "text": text,
#             }
#             payload.append(e)
#         self.data = payload
#
#     def __len__(self):
#         return len(self.data)
#
#     def __getitem__(self, idx):
#         return self.data[idx]
