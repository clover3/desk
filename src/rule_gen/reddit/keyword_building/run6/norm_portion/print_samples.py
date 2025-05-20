import json
from collections import defaultdict

from chair.tab_print import print_table
from rule_gen.reddit.path_helper import get_rp_path


def main():
    annot_path = get_rp_path("clustering", f"100_annot.json")
    target = "personal attack"
    j = json.load(open(annot_path))
    print(len(j))
    table = []
    for e in j:
        name = e["name"]
        any_match = False
        if target in name:
            row = [e["cluster_no"]]
            terms = e["terms10"] + e["terms20"]
            row.append("; ".join(terms))
            table.append(row)
    print_table(table)


if __name__ == "__main__":
    main()