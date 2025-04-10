from rule_gen.reddit.single_run.run_gnq2 import apply_gnq2_to_sb_splits


def main():
    dataset_fmt = "{}_2_val_100"
    run_name_itr = [f"llama_v2_gnq2_{rule_idx}" for rule_idx in range(18)]
    apply_gnq2_to_sb_splits(run_name_itr, dataset_fmt, sb_split="train")


if __name__ == "__main__":
    main()

