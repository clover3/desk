
from rule_gen.cpath import output_root_path
import os

import requests

from rule_gen.cpath import data_root_path


def main():
    list_save_path = os.path.join(output_root_path, "reddit", "popular_list", "overlap_list.txt")
    name_list = [l.strip() for l in open(list_save_path)]
    for sb in name_list:
        url = f"https://www.reddit.com/r/{sb}/"
        response = requests.get(url)
        html_save = os.path.join(
            data_root_path, "reddit", "pages_2025", f"{sb}.html")
        open(html_save, "wb").write(response.content)



    return NotImplemented


if __name__ == "__main__":
    main()