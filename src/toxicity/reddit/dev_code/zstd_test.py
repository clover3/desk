import json
import os

import pyzstd

from chair.misc_lib import make_parent_exists
from toxicity.cpath import output_root_path
from toxicity.reddit.path_helper import load_subreddit_list


class StreamHandler:
    def __init__(self, name):
        save_path = os.path.join(output_root_path, "reddit", f"{name}.jsonl")
        make_parent_exists(save_path)
        self.filename = save_path
        self.line_count = 0
        self.limit = 5000
        self.file = None

        # If the file already exists, count the lines
        if os.path.exists(self.filename):
            with open(self.filename, 'r') as f:
                self.line_count = sum(1 for _ in f)

        # Open the file in append mode
        self.file = open(self.filename, 'ab')

    def add(self, line):
        if self.line_count < self.limit:
            self.file.write(line)
            self.file.flush()  # Ensure data is written to disk
            self.line_count += 1

    def __del__(self):
        if self.file:
            self.file.close()


def main():
    subreddit_list = load_subreddit_list()
    per_subreddit = {k: StreamHandler(k) for k in subreddit_list}
    print(subreddit_list)
    input_file_path = r"C:\Users\leste\Downloads\torrent\reddit\comments\RC_2016-05.zst"
    # with pyzstd.open(input_file_path, "r") as f:
    #     for line in f:
    #         print(line)
    #         break
    output_file_path = r"C:\Users\leste\Downloads\torrent\reddit\comments\RC_2016-05.jsonl"
    # decompress an input file, and write to an output file.
    PARAMS = {pyzstd.DParameter.windowLogMax: 31}
    seen_reddit = set()
    matched_subreddit = 0
    with pyzstd.ZstdFile(input_file_path, "r", level_or_option=PARAMS) as source:
        for line in source:
            j = json.loads(line)
            sb = j['subreddit']
            seen_reddit.add(sb)
            if sb in per_subreddit:
                per_subreddit[sb].add(line)
                if per_subreddit[sb].line_count == 1:
                    matched_subreddit += 1
                    print("matched_subreddit", matched_subreddit, sb)
                if per_subreddit[sb].line_count == per_subreddit[sb].limit - 1:
                    print(f"{sb} has now {per_subreddit[sb].line_count}")

        print("Last", source.tell())


if __name__ == "__main__":
    main()
