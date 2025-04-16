import json
import os
import random
from collections import Counter

import pyzstd

from chair.misc_lib import TELI
from rule_gen.cpath import output_root_path
from rule_gen.reddit.dataset_build.sample_posts import SubredditSampler
from rule_gen.reddit.path_helper import load_subreddit_list


def main():
    subreddit_list = load_subreddit_list()
    input_file_path = os.path.join(output_root_path, "reddit", "dump", "RC_2016-05.zst")
    def line_itr():
        n_read = 0
        with pyzstd.ZstdFile(input_file_path, "r", level_or_option=PARAMS) as source:
            for line in TELI(source, n_lines_maybe):
                yield line
                n_read += 1

        print(f"{n_read} lines read")

    save_dir = os.path.join(output_root_path, "reddit", "subreddit_samples")
    sampler = SubredditSampler(
        subreddit_list, output_dir=save_dir, sample_size=200000)
    PARAMS = {pyzstd.DParameter.windowLogMax: 31}
    n_lines_maybe = 23285177

    print("Counting subreddit occurrences...")
    sampler.count_occurrences(line_itr())

    print("Sampling indices...")
    sampler.sample_indices()

    print("Collecting samples...")
    sampler.collect_samples(line_itr())

    print("Writing samples to disk...")
    sampler.write_samples_to_disk()
    print("Sampling complete.")


if __name__ == "__main__":
    main()