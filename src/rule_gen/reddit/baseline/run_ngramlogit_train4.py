import fire

from rule_gen.reddit.baseline.run_gram_logit_ import run_ngramlogit_train


def main(sb="TwoXChromosomes"):
    data_name = "train_data4"
    save_dir_name = "ngram_logit4"
    print("Running ngram logit")
    print("Tokenizing")
    run_ngramlogit_train(data_name, save_dir_name, sb)


if __name__ == "__main__":
    fire.Fire(main)
