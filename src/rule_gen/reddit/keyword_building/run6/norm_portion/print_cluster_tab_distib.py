import json
from collections import defaultdict

from rule_gen.reddit.path_helper import get_rp_path


def main():
    categories = ["topic", "personal attack", "hate speech", "url", "artifact", "markdown", "special character"]
    annot_path = get_rp_path("clustering", f"100_annot.json")
    j = json.load(open(annot_path))
    print(len(j))
    miss_cnt = 0
    grouped = defaultdict(list)
    for e in j:
        name = e["name"]
        any_match = False
        for c in categories:
            if c in name:
                grouped[c].append(name)
                any_match = True
        if not any_match:
            print(name)

    print("-----")
    for k, v in grouped.items():
        print(k, len(v))


if __name__ == "__main__":
    main()