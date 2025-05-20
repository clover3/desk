import json
import json
import os
import random

from desk_util.io_helper import read_csv
from rule_gen.cpath import output_root_path

exclude_pattern = [
    "This comment has been removed",
    "thank you for your submission",
    "your submission has been removed",
    "I am a bot whose sole purpose is to improve the timeliness ",
    "model",
    "Your thread was removed under",
    "your post has been removed",
    "It will still be visible in the subreddit nonetheless",
    "plus sources that describe the law and explain why it's controversial",
    "How old is your reddit account",
    "your submission was removed for",
    "Thank you for your participation",
    "moderator has removed it according",
    "This submission has been removed",
    "Your submission was removed from",
]

def main():
    text_path = os.path.join(output_root_path, "reddit", "subset", "mod.csv")
    data = read_csv(text_path)
    random.seed(42)
    random.shuffle(data)

    text_list = []
    for sb, text, label_s in data:
        skip = False
        for p in exclude_pattern:
            if p.lower() in text.lower():
                skip = True
                break

        if skip:
            continue
        if int(label_s[0]) == 1:
            text_list.append(text)

    label_name = "pos"
    neg_path = os.path.join(output_root_path, "reddit", "subset", f"mod_{label_name}.csv")
    json.dump(text_list[:100], open(neg_path, "w"), indent=4)


if __name__ == "__main__":
    main()
