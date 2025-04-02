import sys
import json


def main():
    p = sys.argv[1]
    j = json.load(open(p))

    max_text_len = 0
    print("len", len(j))
    for i, e in enumerate(j):
        text = e['input']
        if len(text) > max_text_len:
            max_text_len = len(text)
            print("max text len is", max_text_len, " at ", i)
            # print(text)


if __name__ == "__main__":
    main()