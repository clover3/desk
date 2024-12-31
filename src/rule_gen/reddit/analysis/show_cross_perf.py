from rule_gen.reddit.corpus_sim.compute_sim import get_most_sim
import os

from taskman_client.named_number_proxy import NamedNumberProxy
from rule_gen.cpath import output_root_path
from desk_util.io_helper import save_csv
from rule_gen.reddit.path_helper import get_split_subreddit_list


def build_table(rows, columns, cell_getter):
    table = []
    for row in rows:
        table_row = [row]
        for column in columns:
            cell = cell_getter(row, column)
            table_row.append(cell)
        table.append(table_row)
    return table



def build_save_in_domain_api_cross():
    subreddit_list = get_split_subreddit_list("train")
    search = NamedNumberProxy()

    def get_number(sb, col):
        if col == "inst":
            model_name = f"api_{sb}_detail"
            condition = f"{sb}_val_100"
        elif col == "in-domain":
            model_name = f"bert_{sb}"
            condition = f"{sb}_val"
        elif col == "cross":
            most_sim = get_most_sim(sb, subreddit_list)
            model_name = f"sim_bert_{most_sim}"
            condition = f"{sb}_val"
        else:
            raise ValueError()
        ret = search.get_number(model_name, "f1", condition=condition)
        return ret

    columns = ["in-domain", "inst", "cross"]
    rows = subreddit_list
    table = build_table(rows, columns, get_number)
    save_path = os.path.join(output_root_path, "reddit", "api_f1.csv")
    save_csv(table, save_path)


def main():
    build_save_in_domain_api_cross()


if __name__ == "__main__":
    main()