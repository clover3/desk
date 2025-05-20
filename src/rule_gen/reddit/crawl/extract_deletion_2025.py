from rule_gen.reddit.crawl.extract_deletion_2024_2 import extract_deletion_only_remove_inner
from rule_gen.cpath import output_root_path
import os

def main():
    comment_db_path = "outputs/reddit/db/march3/comments.sqlite"
    deleted_comment_db_path = "outputs/reddit/db/march3/deleted_new.sqlite"
    save_path = os.path.join(output_root_path, "reddit", "2025")
    extract_deletion_only_remove_inner(comment_db_path, deleted_comment_db_path, save_path)



if __name__ == '__main__':
    main()