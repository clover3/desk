import fire

from chair.tab_print import print_table
from taskman_client.named_number_proxy import NamedNumberProxy
from rule_gen.reddit.path_helper import get_split_subreddit_list


def print_perf(dataset_fmt, run_name_fmt, split):
    subreddit_list = get_split_subreddit_list(split)
    search = NamedNumberProxy()
    output = []
    for sb in subreddit_list:
        model_name = run_name_fmt.format(sb)
        dataset = dataset_fmt.format(sb)
        ret = search.get_number(model_name, "f1", condition=dataset)
        row = [model_name, ret]
        output.append(row)
    print_table(output)


def main(run_name_fmt, split="val", dataset_fmt="{}_val_100"):
    # run_name_fmt = "bert_train_mix3"
    print("run_name_fmt", run_name_fmt)
    print("dataset_fmt", dataset_fmt)
    print_perf(dataset_fmt, run_name_fmt, split)


if __name__ == "__main__":
    fire.Fire(main)
