import json
import os
import sqlite3
from datetime import datetime
from typing import Iterable, Dict
from chair.misc_lib import is_power_of_ten

from rule_gen.cpath import output_root_path


def get_time_from_ts(ts):
    TIME_FORMAT = '%Y-%m-%d %H:%M:%S'
    dt = datetime.fromtimestamp(ts)
    return str(dt)


def enum_parsed_dump():
    save_name = "RC_2024-12.zst"
    src_path = os.path.join(output_root_path, "reddit", "dump", f"{save_name}_filtered.zst")
    cnt = 0

    with open(src_path, "rb") as f_in:
        for line in f_in:
            j = json.loads(line)
            yield j
            cnt += 1
            if is_power_of_ten(cnt):
                print("dump iter", cnt)
                print( get_time_from_ts(j['created']))


def read_db(read_path):
    read_conn = sqlite3.connect(read_path)
    last_timestamp = 1733011200 - 1
    query = """
        SELECT *
        FROM comments
        WHERE created_utc > ?
        ORDER BY created_utc ASC
    """
    cursor = read_conn.execute(query, (last_timestamp,))
    cnt = 0
    for comment in cursor:
        comment_dict = {
            'id': comment[0],
            'subreddit': comment[1],
            'created_utc': comment[2],
            'author': comment[3],
            'score': comment[4],
            'body': comment[5],
            'post_id': comment[7],
            'parent_id': comment[8],
            'is_submitter': comment[9],
            'processed_at': comment[10],
        }
        comment_dict['created'] = comment_dict['created_utc']
        yield comment_dict
        cnt += 1
        if is_power_of_ten(cnt):
            print("db iter", cnt)


def group_by_created(iterator):
    prev_key = None
    cur_group = []
    for e in iterator:
        key = e['created']
        if prev_key is None:
            prev_key = key
            cur_group.append(e)
        elif prev_key != key:
            yield cur_group
            cur_group = []
            prev_key = key
            cur_group.append(e)
        else:
            cur_group.append(e)

    if cur_group:
        yield cur_group


class DBwriter:
    def __init__(self, write_path):
        self.write_path = write_path
        self.setup_database()
        self.write_conn = sqlite3.connect(write_path)
        self.cnt = 0

    def setup_database(self):
        """Initialize the deleted_comments table."""
        with sqlite3.connect(self.write_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS deleted_comments (
                    comment_id TEXT PRIMARY KEY,
                    subreddit TEXT NOT NULL,
                    created_utc INTEGER NOT NULL,
                    author TEXT,
                    score INTEGER,
                    body TEXT,
                    post_id TEXT,
                    parent_id TEXT,
                    is_submitter INTEGER,
                    deletion_type TEXT,  -- '[deleted]' or '[removed]'
                    processed_at INTEGER NOT NULL
                )
            """)

            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_deleted_comments_created_utc 
                ON deleted_comments(created_utc)
            """)

            conn.commit()
    def write(self, comment_dict, deletion_type):
        self.write_conn.execute("""
                INSERT OR REPLACE INTO deleted_comments (
                    comment_id, subreddit, created_utc, author, score,
                    body, post_id, parent_id, is_submitter,
                    deletion_type, processed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
            comment_dict['id'], comment_dict['subreddit'],
            comment_dict['created_utc'], comment_dict['author'],
            comment_dict['score'], comment_dict['body'],
            comment_dict['post_id'], comment_dict['parent_id'],
            comment_dict['is_submitter'], deletion_type,
            comment_dict['processed_at']
        ))
        self.write_conn.commit()
        self.cnt += 1
        if is_power_of_ten(self.cnt):
            print("DB write", self.cnt)


def compare_iter(itr1: Iterable[Dict], itr2: Iterable[Dict]):
    iter1 = iter(itr1)
    iter2 = iter(itr2)
    writer = DBwriter("outputs/reddit/db/march3/deleted_new.sqlite")

    group2 = group_by_created(iter2)
    g2_itr = iter(group2)
    g2 = next(g2_itr, None)

    for e in iter1:
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
            e2_d = {e2['id']: e2 for e2 in g2}
            if e['id'] in e2_d:
                e2 = e2_d[e['id']]
                txt = e2["body"]
                if txt.startswith("[removed]"):
                    writer.write(e, "removed")
                elif txt.startswith("[deleted]"):
                    writer.write(e, "deleted")
                else:
                    writer.write(e, "valid")
            else:
                writer.write(e, "not_found")
        else:
            writer.write(e, "not_found")


def main():
    itr1 = read_db("outputs/reddit/db/march3/comments.sqlite")
    itr2 = enum_parsed_dump()
    compare_iter(itr1, itr2)


if __name__ == "__main__":
    main()
