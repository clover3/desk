import os
import sqlite3
import logging
from datetime import datetime
from typing import Dict, Any

import praw
from prawcore import ResponseException
from rule_gen.reddit.path_helper import get_reddit_db_path
import time
from collections import deque


class RateLimiter:
    def __init__(self, max_requests=600, time_window=600):
        """
        Initialize rate limiter with specified constraints

        Args:
            max_requests (int): Maximum number of requests allowed in the time window
            time_window (int): Time window in seconds
        """
        self.max_requests = max_requests
        self.time_window = time_window
        self.requests = deque()  # Store timestamps of requests

    def acquire(self):
        """
        Check if a new request can be made and wait if necessary.
        Returns when it's safe to make a new request.
        """
        current_time = time.time()

        # Remove timestamps older than the time window
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()

        # If we're at the limit, calculate wait time
        if len(self.requests) >= self.max_requests:
            # Calculate how long to wait for the oldest request to expire
            wait_time = self.requests[0] + self.time_window - current_time
            if wait_time > 0:
                print(f"Rate limit reached. Waiting {wait_time:.2f} seconds...")
                time.sleep(wait_time)
                # After waiting, clean up old timestamps again
                current_time = time.time()
                while self.requests and current_time - self.requests[0] >= self.time_window:
                    self.requests.popleft()

        # Add current request timestamp
        self.requests.append(current_time)

    def get_current_usage(self):
        """
        Get current rate limit usage statistics

        Returns:
            dict: Contains current usage information
        """
        current_time = time.time()

        # Clean up old timestamps
        while self.requests and current_time - self.requests[0] >= self.time_window:
            self.requests.popleft()

        return {
            'current_requests': len(self.requests),
            'max_requests': self.max_requests,
            'time_window': self.time_window,
            'remaining_requests': self.max_requests - len(self.requests)
        }


class DeletedCommentProcessor:
    def __init__(self, read_path: str, write_path: str, batch_size: int = 100,
                 max_retries: int = 10, initial_retry_delay: int = 60,
                 run_time=3600 * 24, threshold: int = 10000):
        self.read_path = read_path
        self.write_path = write_path
        self.batch_size = batch_size
        self.reddit = praw.Reddit()
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay
        self.start = time.time()
        self.run_time = run_time
        self.threshold = threshold
        self.exclude_subreddit_list = set()  # Using set for O(1) lookups
        self.subreddit_counts = {}  # Dictionary to store counts

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.setup_database()
        self.load_subreddit_counts()


    def load_subreddit_counts(self):
        """Load current subreddit counts from database and initialize exclusion list."""
        with sqlite3.connect(self.write_path) as conn:
            cursor = conn.execute("""
                SELECT subreddit, COUNT(*) as count
                FROM deleted_comments
                GROUP BY subreddit
            """)

            for subreddit, count in cursor.fetchall():
                self.subreddit_counts[subreddit] = count
                if count >= self.threshold:
                    self.exclude_subreddit_list.add(subreddit)
                    self.logger.info(f"Excluding subreddit {subreddit} with {count} comments")

    def update_subreddit_count(self, subreddit: str):
        """Update count for a subreddit and check if it should be excluded."""
        self.subreddit_counts[subreddit] = self.subreddit_counts.get(subreddit, 0) + 1

        if (self.subreddit_counts[subreddit] >= self.threshold and
                subreddit not in self.exclude_subreddit_list):
            self.exclude_subreddit_list.add(subreddit)
            self.logger.info(f"Adding {subreddit} to exclusion list - reached {self.threshold} comments")

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

    def get_last_processed_timestamp(self) -> int:
        """Get the timestamp of the last processed comment."""
        with sqlite3.connect(self.write_path) as conn:
            cursor = conn.execute("""
                SELECT created_utc, datetime(created_utc, 'unixepoch')
                FROM deleted_comments
                ORDER BY created_utc DESC
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                self.logger.info(f"Resuming from timestamp: {result[1]}")
                return result[0]
            return 0

    def check_deleted(self, comment: Dict[str, Any]) -> tuple[bool, str]:
        """Check if comment is deleted and return deletion type."""
        cid = comment["id"]
        retry_count = 0

        while True:
            try:
                comment = praw.models.Comment(self.reddit, id=cid)
                txt = comment.body

                if txt.startswith("[removed]"):
                    return True, "removed"
                elif txt.startswith("[deleted]"):
                    return True, "deleted"
                return False, ""

            except ResponseException as e:
                if hasattr(e, 'response') and e.response.status_code == 429:
                    if not self.handle_429_error(retry_count):
                        self.logger.error(f"Failed to process comment {cid} after max retries")
                        return False, ""
                    retry_count += 1
                else:
                    self.logger.error(f"Error processing comment {cid}: {str(e)}")
                    return False, ""
            except Exception as e:
                self.logger.error(f"Unexpected error processing comment {cid}: {str(e)}")
                return False, ""


    def handle_429_error(self, retry_count: int) -> bool:
        """
        Handle 429 error with exponential backoff.

        Args:
            retry_count: Current retry attempt number

        Returns:
            bool: True if should retry, False if max retries exceeded
        """
        if retry_count >= self.max_retries:
            self.logger.error(f"Max retries ({self.max_retries}) exceeded for 429 error. Stopping crawl.")
            return False

        # Calculate delay with exponential backoff
        delay = self.initial_retry_delay * (2 ** retry_count)
        self.logger.warning(f"Received 429 error. Attempt {retry_count + 1}/{self.max_retries}. "
                            f"Waiting {delay} seconds before retry...")
        time.sleep(delay)
        return True

    def process_comments(self):
        self.logger.info(f"process_comments")

        """Process comments in batches and save only deleted ones."""
        last_timestamp = self.get_last_processed_timestamp()
        processed_count = 0
        deleted_count = 0
        skipped_count = 0

        try:
            read_conn = sqlite3.connect(self.read_path)
            write_conn = sqlite3.connect(self.write_path)

            while True:
                if time.time() - self.start > self.run_time:
                    self.logger.info("Time limit exceeded...")
                    return

                query = """
                    SELECT *
                    FROM comments
                    WHERE created_utc > ?
                    ORDER BY created_utc
                    LIMIT ?
                """

                cursor = read_conn.execute(query, (last_timestamp, self.batch_size))
                comments = cursor.fetchall()

                if not comments:
                    break

                current_time = int(datetime.now().timestamp())

                for comment in comments:
                    comment_dict = {
                        'id': comment[0],
                        'subreddit': comment[1],
                        'created_utc': comment[2],
                        'author': comment[3],
                        'score': comment[4],
                        'body': comment[5],
                        'post_id': comment[7],
                        'parent_id': comment[8],
                        'is_submitter': comment[9]
                    }
                    if comment_dict['subreddit'] in self.exclude_subreddit_list:
                        skipped_count += 1
                        last_timestamp = comment_dict['created_utc']
                        processed_count += 1
                        continue

                    is_deleted, deletion_type = self.check_deleted(comment_dict)

                    if is_deleted:
                        write_conn.execute("""
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
                            current_time
                        ))
                        deleted_count += 1

                        write_conn.commit()

                    last_timestamp = comment_dict['created_utc']
                    processed_count += 1

                    if processed_count % self.batch_size == 0:
                        self.logger.info(
                            f"Processed {processed_count} comments, found {deleted_count} deleted, "
                            f"up to {datetime.fromtimestamp(last_timestamp)}")

        except KeyboardInterrupt:
            self.logger.info("Gracefully shutting down...")
            write_conn.commit()
        except Exception as e:
            self.logger.error(f"Processing error: {e}")
            write_conn.rollback()
            raise e
        finally:
            read_conn.close()
            write_conn.close()
            self.logger.info(
                f"Processing completed. Total comments processed: {processed_count}, "
                f"Deleted comments found: {deleted_count}, "
                f"Skipped comments: {skipped_count}")

    def get_excluded_subreddits_stats(self):
        """Get statistics about excluded subreddits."""
        self.logger.info("\nExcluded Subreddits Statistics:")
        for subreddit in sorted(self.exclude_subreddit_list):
            count = self.subreddit_counts.get(subreddit, 0)
            self.logger.info(f"- {subreddit}: {count:,} comments")

    def get_deletion_stats(self):
        """Get statistics about deleted comments."""
        with sqlite3.connect(self.write_path) as conn:
            cursor = conn.execute("""
                SELECT 
                    COUNT(*) as total_deleted,
                    MIN(datetime(created_utc, 'unixepoch')) as earliest_comment,
                    MAX(datetime(created_utc, 'unixepoch')) as latest_comment,
                    SUM(CASE WHEN deletion_type = 'removed' THEN 1 ELSE 0 END) as removed_count,
                    SUM(CASE WHEN deletion_type = 'deleted' THEN 1 ELSE 0 END) as deleted_count
                FROM deleted_comments
            """)
            stats = cursor.fetchone()

            if stats:
                self.logger.info(f"""
                Deletion Statistics:
                - Total Deleted Comments: {stats[0]:,}
                - Date Range: {stats[1]} to {stats[2]}
                - Removed: {stats[3]:,}
                - Deleted: {stats[4]:,}
                """)

            return stats


def main(batch_size: int = 100):
    read_path = os.path.join(get_reddit_db_path(), "comments.sqlite")
    write_path = os.path.join(get_reddit_db_path(), "deleted_comments.sqlite")

    processor = DeletedCommentProcessor(read_path, write_path, batch_size)
    # processor.process_comments()
    processor.get_deletion_stats()


if __name__ == "__main__":
    import fire
    fire.Fire(main)