import json
from collections import defaultdict

from chair.misc_lib import exist_or_mkdir
from rule_gen.cpath import output_root_path
import os
from rule_gen.reddit.crawl.compute_deletion_rate import read_deleted_comments_db
from rule_gen.reddit.handle_dump.db_helper import read_db, group_by_created


def main():
    itr1 = read_db("outputs/reddit/db/2024/comments.sqlite")
    read_path = "outputs/reddit/db/2024/deleted_comments.sqlite"
    itr2 =  read_deleted_comments_db(read_path)
    group1 = group_by_created(itr1)
    group2 = group_by_created(itr2)
    g2_itr = iter(group2)
    g2 = next(g2_itr, None)

    all_items = defaultdict(list)
    p = os.path.join(output_root_path, "reddit", "2024data")
    exist_or_mkdir(p)

    try:
        for g1 in group1:
            for e in g1:
                sb1 = e['subreddit']
                key1 = e['created']
                match = False
                if g2 is None:
                    match = False
                else:
                    key2 = g2[0]["created"]
                    while key2 < key1:
                        g2 = next(g2_itr, None)
                        key2 = g2[0]["created"]

                    if key1 == key2:
                        match = True
                    elif key1 < key2:
                        match = False
                    else:
                        pass
                if match:
                    id1 = e['id']
                    e2_d = {e2['comment_id']: e2 for e2 in g2}
                    if id1 in e2_d:
                        e2 = e2_d[id1]
                        txt = e2["body"]
                        sb2 = e2["subreddit"]
                        assert sb1 == sb2
                        all_items[sb1, "pos"].append(txt)
                else:
                    all_items[sb1, "neg"].append(e["body"])
    except TypeError:
        pass

    for key, entries in all_items.items():
        sb, label = key
        label_i = {
            "pos": 1,
            "neg": 0,
        }[label]
        save_path = os.path.join(p, f"{sb}-{label}.jsonl")
        with open(save_path, "w") as f:
            for e in entries:
                f.write(json.dumps({"text": e, "label": label_i}) + "\n")


if __name__ == '__main__':
    main()