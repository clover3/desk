from rule_gen.reddit.handle_dump.compare_dumps_and_db import read_db
from datetime import datetime


def main():
    itr1 = read_db("outputs/reddit/db/march3/comments.sqlite")
    cnt = 0
    for e in itr1:
        if cnt % 100000 == 0:
            st = e['created']
            ed = e['processed_at']
            gap = ed - st
            print(gap, datetime.fromtimestamp(st))

        cnt += 1


if __name__ == "__main__":
    main()
