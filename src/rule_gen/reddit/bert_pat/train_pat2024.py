import fire

from rule_gen.reddit.bert_pat.train_pat import reddit_train_pat_exp_inner


def reddit_train_pat_exp(sb="TwoXChromosomes", debug=False):
    model_name = f"bert_ts_2024_{sb}"
    data_name = "2024"

    reddit_train_pat_exp_inner(data_name, debug, model_name, sb)


if __name__ == "__main__":
    fire.Fire(reddit_train_pat_exp)
