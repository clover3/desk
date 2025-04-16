from rule_gen.reddit.dataset_build2.train_comb3 import build_sampled_train_dataset


if __name__ == "__main__":
    src_data_name = "train_data3"

    build_sampled_train_dataset(
        "train_data3",
        "train_comb5",
        1000,
        2000)
