import os

from toxicity.cpath import output_root_path


def get_open_ai_mod_csv_path(role):
    return os.path.join(output_root_path, "open_ai_mod", f"{role}.csv")
