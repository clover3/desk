import json

from rule_gen.reddit.path_helper import get_rp_path


def main():
    annot_path = get_rp_path("clustering", f"100_annot.json")
    j = json.load(open(annot_path))
    print(len(j))
    miss_cnt = 0
    for e in j:
        if e["name"] == "X":
            miss_cnt += 1

    print(miss_cnt)

if __name__ == "__main__":
    main()