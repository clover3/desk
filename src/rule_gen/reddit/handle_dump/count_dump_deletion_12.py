import json
import os
from collections import Counter

from rule_gen.cpath import output_root_path


def main():
    input_file_path = os.path.join(output_root_path, "reddit", "dump", "RC_2024-12.zst_filtered.zst")
    cnt = 0
    counter = Counter()
    with open(input_file_path, "r") as source:
        source.readline()
        for line in source:
            post = json.loads(line)
            created_utc = post["created_utc"]
            txt = post["body"]
            if txt.startswith("[removed]"):
                counter["[removed]"] += 1
            elif txt.startswith("[deleted]"):
                counter["[deleted]"] += 1
            cnt += 1
    print(created_utc)
    print(counter)
    print(cnt)

if __name__ == "__main__":
    main()