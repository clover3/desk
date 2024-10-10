import json
import os
import random
from collections import Counter

import pyzstd

from chair.misc_lib import TELI, make_parent_exists
from toxicity.cpath import output_root_path, data_root_path
from toxicity.path_helper import load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    subreddits = set(subreddit_list)
    input_file_path = os.path.join(data_root_path, "reddit", "dump", "RC_2016-05.zst")
    save_path = os.path.join(output_root_path, "reddit", "dump", "RC_2016-05_filtered.zst")
    make_parent_exists(save_path)
    PARAMS = {pyzstd.DParameter.windowLogMax: 31}
    n_lines_maybe = 67108864

    def line_itr():
        with pyzstd.ZstdFile(input_file_path, "r", level_or_option=PARAMS) as source:
            for line in TELI(source, n_lines_maybe):
                yield line

    f = open(save_path, "wb")
    for i, line in enumerate(line_itr()):
        post = json.loads(line)
        subreddit = post.get('subreddit')
        if subreddit in subreddits:
            f.write(line)


if __name__ == "__main__":
    main()