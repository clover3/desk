import json

from rule_gen.cpath import output_root_path
import os

from rule_gen.reddit.crawl.find_new_big_reddit.check_created_one_by_one import cut_line
from rule_gen.reddit.crawl.find_new_big_reddit.get_popular_sblist import read_existing_jsonl


def main():
    file_name = "sfw_subreddits.jsonl"
    output_file = os.path.join(output_root_path, "reddit", "popular_list", file_name)
    existing_data, popular_sbs = read_existing_jsonl(output_file)
    print("Existing data size: ", len(existing_data))
    print("Existing names size: ", len(popular_sbs))
    june2024 = cut_line()

    file_name = "new_sb_list.jsonl"
    new_sb_path = os.path.join(output_root_path, "reddit", "popular_list", file_name)
    list_save_path = os.path.join(output_root_path, "reddit", "popular_list", "overlap_list.txt")

    f_w = open(list_save_path, "w")
    old_cnt = 0
    for line in open(new_sb_path, "r"):
        j = json.loads(line)
        if j['created_utc'] < june2024:
            old_cnt += 1
        else:
            if j['name'] in popular_sbs:
                print(j)
                f_w.write(j['name'] + "\n")

    print(old_cnt)


if __name__ == "__main__":
    main()