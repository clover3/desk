import sqlite3

from chair.misc_lib import is_power_of_ten


def read_db(read_path, last_timestamp =0):
    read_conn = sqlite3.connect(read_path)
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
