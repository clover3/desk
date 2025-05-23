import json
import os
import sys
from typing import Iterable

import pyzstd

from chair.misc_lib import TELI, make_parent_exists
from rule_gen.cpath import output_root_path, data_root_path
from rule_gen.reddit.path_helper import load_subreddit_list

def guess_line_by_size(input_file_path):
    byte_per_line = 100
    return int(os.path.getsize(input_file_path) / byte_per_line)


def filter_dump_by_subreddits(input_file_path, save_path, subreddits):
    make_parent_exists(save_path)
    PARAMS = {pyzstd.DParameter.windowLogMax: 31}
    n_lines_maybe = guess_line_by_size(input_file_path)
    def line_itr():
        n_read = 0
        with pyzstd.ZstdFile(input_file_path, "r", level_or_option=PARAMS) as source:
            for line in TELI(source, n_lines_maybe):
                yield line
                n_read += 1

        print(f"{n_read} lines read")

    f = open(save_path, "wb")
    for i, line in enumerate(line_itr()):
        post = json.loads(line)

        subreddit = post.get('subreddit')
        if subreddit in subreddits:
            f.write(line)






def main():
    # 304008295
    subreddit_list = load_subreddit_list()
    subreddits = set(subreddit_list)
    input_file_path = sys.argv[1]
    save_name = os.path.basename(input_file_path)
    save_path = os.path.join(output_root_path, "reddit", "dump", f"{save_name}_filtered.zst")
    filter_dump_by_subreddits(input_file_path, save_path, subreddits)



if __name__ == "__main__":
    main()