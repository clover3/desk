from chair.misc_lib import is_power_of_ten
from rule_gen.cpath import output_root_path
import os
import json


def main():
    save_name = "RC_2024-12.zst"
    src_path = os.path.join(output_root_path, "reddit", "dump", f"{save_name}_filtered.zst")
    prev_created = None
    cnt = 0
    with open(src_path, "rb") as f_in:
        for line in f_in:
            cnt += 1
            if is_power_of_ten(cnt):
                print(cnt)
            j = json.loads(line)
            created = j["created"]

            if prev_created is not None:
                if created <= prev_created:
                    print("No")
                prev_created = created


if __name__ == "__main__":
    main()