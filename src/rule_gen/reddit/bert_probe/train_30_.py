from desk_util.io_helper import init_logging
from rule_gen.reddit.bert_probe.train_bert_probe import do_train_bert_probe
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    init_logging()
    for sb in get_split_subreddit_list("train")[30:]:
        try:
            do_train_bert_probe(sb)
        except Exception as e:
            print(e)


if __name__ == "__main__":
    main()
