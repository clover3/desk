import sys
import json


def main():
    p = sys.argv[1]
    j = json.load(open(p))
    print("len", len(j))
    j = j[1874:] + j[:1874]
    save_p = p + "_mod.json"
    json.dump(j, open(save_p, "w"), indent=4)


if __name__ == "__main__":
    main()