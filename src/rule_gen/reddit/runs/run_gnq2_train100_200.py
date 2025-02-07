from rule_gen.reddit.runs.run_gnq2 import apply_gnq2_to_sb_splits


def main():
    dataset_fmt = "{}_2_train_100_200"
    apply_gnq2_to_sb_splits(dataset_fmt, sb_split="train")


if __name__ == "__main__":
    main()

