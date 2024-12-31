from sklearn.model_selection import train_test_split
from datasets import load_dataset

from desk_util.io_helper import save_csv
from desk_util.path_helper import get_cola_train_data_path


def save_data():
    dataset = load_dataset("glue", "mrpc")
    dataset = dataset["train"]  # Just take the training split for now
    dataset = [(e["sentence"], e["label"]) for e in dataset]

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    def save(data, role):
        save_csv(data, get_cola_train_data_path(role))

    save(train_data, "train")
    save(test_data, "val")
