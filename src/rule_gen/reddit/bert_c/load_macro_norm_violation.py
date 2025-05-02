from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path
import os


def get_macro_norm_listing_path():
    return os.path.join(
        output_root_path, "reddit", "macro-norm-list.csv")


def get_macro_norm_dir_path():
    return os.path.join(
        output_root_path, "reddit", "macro-norm-violations")



def load_all_norm_data() -> list[tuple[str, list[str]]]:
    # Each have 5059 items
    name_to_filename = read_csv(get_macro_norm_listing_path())
    root_dir = get_macro_norm_dir_path()
    output = []
    for name, file_name in name_to_filename:
        p = os.path.join(root_dir, file_name)
        entries = read_csv(p)
        text_list = []
        for e in entries:
            if len(e) != 1:
                print("Not all rows are single column")
            text_list.append(e[0])

        output.append((name, text_list))
    return output




def load_all_norm_data_splits():
    # Each have 5059 items
    name_to_filename = read_csv(get_macro_norm_listing_path())
    root_dir = get_macro_norm_dir_path()
    splits = {"train": [], "val": [], "test": []}
    for name, file_name in name_to_filename:
        p = os.path.join(root_dir, file_name)
        entries = read_csv(p)
        text_list = []
        for e in entries:
            if len(e) != 1:
                print("Not all rows are single column")
            text_list.append(e[0])

        splits["train"].append((name, text_list[:3000]))
        splits["val"].append((name, text_list[3000:4000]))
        splits["test"].append((name, text_list[4000:5000]))
    return splits


def load_norm_id_mapping():
    d = {}
    norm_id = 1
    name_to_filename = read_csv(get_macro_norm_listing_path())
    for name, file_name in name_to_filename:
        d[name] = norm_id
        norm_id += 1
    return d


