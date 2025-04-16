import sqlite3
from collections import Counter

from chair.list_lib import left
from chair.tab_print import print_table
from rule_gen.reddit.handle_dump.db_helper import read_db, group_by_created


def is_power_of_ten(n):
    return n > 0 and (n == 1 or 10 ** int(len(str(n)) - 1) == n)

def read_deleted_comments_db(read_path):
    read_conn = sqlite3.connect(read_path)
    last_timestamp = 1733011200 - 1  # Adjust as needed
    last_timestamp = 0
    query = """
        SELECT *
        FROM deleted_comments
        WHERE created_utc > ?
        ORDER BY created_utc ASC
    """
    cursor = read_conn.execute(query, (last_timestamp,))
    cnt = 0
    for comment in cursor:
        comment_dict = {
            'comment_id': comment[0],
            'subreddit': comment[1],
            'created_utc': comment[2],
            'author': comment[3],
            'score': comment[4],
            'body': comment[5],
            'post_id': comment[6],
            'parent_id': comment[7],
            'is_submitter': comment[8],
            'deletion_type': comment[9],  # '[deleted]' or '[removed]'
            'processed_at': comment[10],
        }
        comment_dict['created'] = comment_dict['created_utc']
        yield comment_dict
        cnt += 1
        if is_power_of_ten(cnt):
            print("deleted_comments db iter", cnt)



def main():
    itr1 = read_db("outputs/reddit/db/2024/comments.sqlite")
    read_path = "outputs/reddit/db/2024/deleted_comments.sqlite"
    itr2 =  read_deleted_comments_db(read_path)
    group1 = group_by_created(itr1)
    group2 = group_by_created(itr2)
    g2_itr = iter(group2)
    g2 = next(g2_itr, None)

    counter = Counter()

    try:
        for g1 in group1:
            for e in g1:
                sb1 = e['subreddit']
                counter[(sb1, "all")] += 1
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
                        counter[(sb1, "delete")] += 1
                        delete_cnt = counter[(sb1, "delete")]
                        all_cnt = counter[(sb1, "all")]
                        # if is_power_of_ten(delete_cnt):
                        #     print(sb1, delete_cnt, all_cnt)
    except TypeError:
        pass

    sb_list = left(counter.keys())
    sb_list = list(set(sb_list))
    sb_list.sort()
    table = []
    for sb in sb_list:
        del_rate= counter[sb, "delete"] / counter[sb, "all"]
        row = [sb, counter[sb, "delete"], counter[sb, "all"], del_rate]
        table.append(row)
    print_table(table)



if __name__ == "__main__":
    main()
