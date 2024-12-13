import json
import os
import random
from collections import Counter

import pyzstd

from chair.misc_lib import TELI
from toxicity.cpath import output_root_path, data_root_path
from toxicity.reddit.path_helper import load_subreddit_list


class SubredditSampler:
    def __init__(self, subreddits, sample_size=10000, output_dir='output'):
        self.subreddits = set(subreddits)
        self.sample_size = sample_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        self.subreddit_counts = Counter()
        self.sampled_indices = {}
        self.samples = {subreddit: [] for subreddit in subreddits}

    def count_occurrences(self, line_itr):
        for i, line in enumerate(line_itr):
            post = json.loads(line)
            subreddit = post.get('subreddit')
            if subreddit in self.subreddits:
                self.subreddit_counts[subreddit] += 1

    def sample_indices(self):
        for subreddit, count in self.subreddit_counts.items():
            if count <= self.sample_size:
                self.sampled_indices[subreddit] = set(range(count))
            else:
                self.sampled_indices[subreddit] = set(random.sample(range(count), self.sample_size))

    def collect_samples(self, line_itr):
        subreddit_counters = {subreddit: 0 for subreddit in self.subreddits}

        for line in line_itr:
            post = json.loads(line)
            subreddit = post.get('subreddit')

            if subreddit in self.subreddits:
                if subreddit_counters[subreddit] in self.sampled_indices[subreddit]:
                    self.samples[subreddit].append(post)
                subreddit_counters[subreddit] += 1

    def write_samples_to_disk(self):
        for subreddit, posts in self.samples.items():
            filename = os.path.join(self.output_dir, f"{subreddit}.jsonl")
            with open(filename, 'w') as f:
                for post in posts:
                    json.dump(post, f)
                    f.write('\n')


def main():
    subreddit_list = load_subreddit_list()
    # input_file_path = os.path.join(data_root_path, "reddit", "dump", "RC_2016-05.zst")
    input_file_path = os.path.join(output_root_path, "reddit", "dump", "RC_2016-05_filtered.jsonl")

    save_dir = os.path.join(output_root_path, "reddit", "subreddit_samples")
    sampler = SubredditSampler(
        subreddit_list, output_dir=save_dir, sample_size=200000)
    PARAMS = {pyzstd.DParameter.windowLogMax: 31}
    n_lines_maybe = 23285177

    def line_itr():
        with open(input_file_path, "r") as source:
            for line in TELI(source, n_lines_maybe):
                yield line

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