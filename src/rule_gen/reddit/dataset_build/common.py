from desk_util.io_helper import save_csv
from desk_util.path_helper import get_csv_dataset_path, get_label_path


def generated_dataset_and_label(data, dataset_name):
    data_w_id = []
    for idx, (text, label) in enumerate(data):
        data_id = f"{dataset_name}_{idx}"
        data_w_id.append((data_id, text, label))
    payload = [(e[0], e[1]) for e in data_w_id]
    labels = [(e[0], e[2]) for e in data_w_id]
    save_path = get_csv_dataset_path(dataset_name)
    save_csv(payload, save_path)
    save_path = get_label_path(dataset_name)
    save_csv(labels, save_path)
