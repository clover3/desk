import fire

from rule_gen.reddit.display.show_prec_recall import print_perf


def main(run_name_fmt, split="val", dataset_fmt="{}_val_100"):
    # run_name_fmt = "bert_train_mix3"
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)
    columns = ["f1"]
    print_perf(dataset_fmt, run_name_fmt, split, columns)


if __name__ == "__main__":
    fire.Fire(main)
