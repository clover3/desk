import os

from taskman_client.named_number_proxy import NamedNumberProxy
from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv
from rule_gen.reddit.path_helper import get_group1_list


def main():
    subreddit_list = get_group1_list()
    search = NamedNumberProxy()
    output = []
    for src in subreddit_list:
        model_name = f"bert2_{src}"
        row = [model_name, ]

        for dst in subreddit_list:
            condition = f"{dst}_2_val_100"
            ret = search.get_number(model_name, "f1", condition)
            row.append(ret)

        output.append(row)

    save_path = os.path.join(output_root_path, "reddit", "group1.csv")
    save_csv(output, save_path)


if __name__ == "__main__":
    main()
