import json
import os
from collections import Counter

from rule_gen.cpath import output_root_path


def main():
    input_file_path = os.path.join(output_root_path, "reddit", "dump", "RC_2024-10_filtered.zst")
    start_time = 1729816052
    end_time = 1730334452
    cnt = 0
    counter = Counter()
    skip = int(os.path.getsize(input_file_path) * 0.7)
    with open(input_file_path, "r") as source:
        source.seek(skip)
        source.readline()
        for line in source:
            post = json.loads(line)
            created_utc = post["created_utc"]
            if created_utc < start_time:
                continue


            txt = post["body"]
            if txt.startswith("[removed]"):
                counter["[removed]"] += 1
            elif txt.startswith("[deleted]"):
                counter["[deleted]"] += 1

            if created_utc > end_time:
                break

            cnt += 1
    print(created_utc)
    print(counter)
    print(cnt)

if __name__ == "__main__":
    main()