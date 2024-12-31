from collections import Counter

from datasets import load_dataset
from sklearn.model_selection import train_test_split

from chair.misc_lib import get_second
from desk_util.io_helper import save_csv, read_csv
from desk_util.path_helper import get_cola_train_data_path, get_model_save_path, \
    get_model_log_save_dir_path
from rule_gen.reddit.train_bert import finetune_bert


def save_data():
    dataset = load_dataset("glue", "cola")
    dataset = dataset["train"]  # Just take the training split for now
    dataset = [(e["sentence"], e["label"]) for e in dataset]

    train_data, test_data = train_test_split(dataset, test_size=0.2, random_state=42)

    def save(data, role):
        save_csv(data, get_cola_train_data_path(role))

    save(train_data, "train")
    save(test_data, "val")


def train():
    model_name = f"bert_cola"
    finetune_bert(
        train_data_path=get_cola_train_data_path("train"),
        eval_data_path=get_cola_train_data_path("val"),
        model_name='bert-base-uncased',
        output_dir=get_model_save_path(model_name),
        final_model_dir=get_model_save_path(model_name),
        logging_dir=get_model_log_save_dir_path(model_name),
        num_train_epochs=3,
        learning_rate=5e-5,
        train_batch_size=4,
        eval_batch_size=64,
        max_length=256,
        num_labels=3,
        warmup_ratio=0.1  # 10% of total steps for warmup
    )


def distrib():
    data = read_csv(get_cola_train_data_path("val"))
    print(Counter(map(get_second, data)))



def main():
    # save_data()
    distrib()


if __name__ == "__main__":
    main()
