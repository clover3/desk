from rule_gen.reddit.dataset_build.common import generated_dataset_and_label
from rule_gen.reddit.transfer.runner.bt2_100 import load_train_mix3_n_item


def main100():
    n_item = 100
    new_dataset_name = "src_mix100"
    data = load_train_mix3_n_item(n_item)
    generated_dataset_and_label(data, new_dataset_name)


if __name__ == "__main__":
    main100()
