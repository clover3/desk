from rule_gen.runner.predict_split3 import predict_sb_split


def main():
    for run_name_fmt in ["chatgpt_{}_both"]:
        predict_sb_split(run_name_fmt, split="val", do_eval=True,
                         dataset_fmt="{}_3_test_1000")


if __name__ == "__main__":
    main()