import os
import time

import fire
import praw
import sqlite3
import logging
from datetime import datetime
from typing import List, Optional, Literal
from pathlib import Path
from prawcore.exceptions import ResponseException
from requests.exceptions import HTTPError

from toxicity.reddit.path_helper import load_subreddit_list, get_reddit_db_path


class RedditCrawler:
    def __init__(self,
                 subreddits: List[str],
                 content_type: Literal['posts', 'comments'],
                 db_dir_path: str,
                 run_time=3600 * 4,
                 max_retries=5,
                 initial_retry_delay=60
                 ):
        """
        Initialize crawler for multiple subreddits in single stream.

        Args:
            subreddits: List of subreddit names without 'r/'
            content_type: Type of content to crawl ('posts' or 'comments')
            db_path: SQLite database path
            run_time: Total run time in seconds
            max_retries: Maximum number of retry attempts for 429 errors
            initial_retry_delay: Initial delay in seconds before retrying
        """
        self.subreddits = subreddits
        self.content_type = content_type
        self.db_path = os.path.join(db_dir_path, f"{content_type}.sqlite")
        self.max_retries = max_retries
        self.initial_retry_delay = initial_retry_delay

        # Combine subreddits with '+' for PRAW
        combined_names = '+'.join(subreddits)

        self.reddit = praw.Reddit()
        self.subreddit = self.reddit.subreddit(combined_names)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        self.setup_database()
        self.run_time = run_time
        self.start = time.time()

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

    def get_stream(self, last_item_id: Optional[str] = None):
        """
        Get the appropriate stream based on content type.

        Args:
            last_item_id: ID of the last processed item

        Returns:
            PRAW stream object
        """
        if self.content_type == 'posts':
            return self.subreddit.stream.submissions(
                pause_after=-1,
                skip_existing=False,
                continue_after_id=last_item_id
            )
        else:  # comments
            return self.subreddit.stream.comments(
                pause_after=-1,
                skip_existing=False,
                continue_after_id=last_item_id
            )

    def crawl(self, batch_size: int = 100):
        """Start crawling all subreddits in a single stream."""
        last_item_id = self.get_last_item_info()

        if last_item_id:
            self.logger.info(f"Resuming crawl after {self.content_type[:-1]} ID: {last_item_id}")
        else:
            self.logger.info(f"Starting new crawl for {self.content_type} in: {', '.join(self.subreddits)}")

        processed_in_session = 0
        subreddit_counts = {sub: 0 for sub in self.subreddits}
        retry_count = 0

        while True:
            try:
                self.logger.info(f"Initialize stream object")
                stream = self.get_stream(last_item_id)

                for item in stream:
                    if item is None:
                        continue

                    subreddit_name = str(item.subreddit)
                    success = self.save_post(item) if self.content_type == 'posts' else self.save_comment(item)

                    if success:
                        processed_in_session += 1
                        subreddit_counts[subreddit_name] = subreddit_counts.get(subreddit_name, 0) + 1
                        last_item_id = item.id  # Update last_item_id for potential retries

                        if processed_in_session % batch_size == 0:
                            self.logger.info(
                                f"Processed {processed_in_session} {self.content_type} in this session. "
                            )

                    if time.time() - self.start > self.run_time:
                        self.logger.info("Time limit exceeded...")
                        return

                # If we get here without errors, reset retry count
                retry_count = 0

            except (ResponseException, HTTPError) as e:
                if hasattr(e, 'response') and e.response.status_code == 429:
                    if not self.handle_429_error(retry_count):
                        break
                    retry_count += 1
                    continue
                else:
                    self.logger.error(f"Unexpected API error: {e}")
                    break

            except KeyboardInterrupt:
                self.logger.info("Gracefully shutting down...")
                break
            except Exception as e:
                self.logger.error(f"Crawling error: {e}")
                break

        # Final statistics
        final_stats = self.get_statistics()
        self.logger.info(f"Crawling stopped. Final {self.content_type} statistics:")
        for subreddit, stats in final_stats.items():
            self.logger.info(f"r/{subreddit}: {stats} {self.content_type}")

    def setup_database(self):
        """Initialize SQLite database with tables for both posts and comments."""
        with sqlite3.connect(self.db_path) as conn:
            # Posts table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS posts (
                    id TEXT PRIMARY KEY,
                    subreddit TEXT NOT NULL,
                    created_utc INTEGER NOT NULL,
                    title TEXT,
                    author TEXT,
                    score INTEGER,
                    url TEXT,
                    selftext TEXT,
                    num_comments INTEGER,
                    permalink TEXT,
                    processed_at INTEGER NOT NULL,
                    UNIQUE(id)
                )
            """)

            # Comments table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS comments (
                    id TEXT PRIMARY KEY,
                    subreddit TEXT NOT NULL,
                    created_utc INTEGER NOT NULL,
                    author TEXT,
                    score INTEGER,
                    body TEXT,
                    permalink TEXT,
                    post_id TEXT,
                    parent_id TEXT,
                    is_submitter INTEGER,
                    processed_at INTEGER NOT NULL,
                    UNIQUE(id)
                )
            """)

            # Indexes for efficient queries
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_posts_subreddit_created 
                ON posts(subreddit, created_utc DESC)
            """)
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_comments_subreddit_created 
                ON comments(subreddit, created_utc DESC)
            """)

            conn.commit()

    def get_last_item_info(self) -> Optional[str]:
        """Get the most recent item ID across all subreddits."""
        table = "posts" if self.content_type == "posts" else "comments"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(f"""
                SELECT id, subreddit, datetime(created_utc, 'unixepoch')
                FROM {table}
                ORDER BY created_utc DESC 
                LIMIT 1
            """)
            result = cursor.fetchone()
            if result:
                self.logger.info(f"Last {self.content_type[:-1]}: {result[0]} from r/{result[1]} at {result[2]}")
                return result[0]
            return None

    def save_post(self, post: praw.models.Submission) -> bool:
        """Save post to database."""
        try:
            current_time = int(datetime.now().timestamp())
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO posts (
                        id, subreddit, created_utc, title, author, score,
                        url, selftext, num_comments, permalink, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    post.id,
                    str(post.subreddit),
                    int(post.created_utc),
                    post.title,
                    str(post.author) if post.author else '[deleted]',
                    post.compute_interaction_score,
                    post.url,
                    post.selftext,
                    post.num_comments,
                    post.permalink,
                    current_time
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving post {post.id} from r/{post.subreddit}: {e}")
            return False

    def save_comment(self, comment: praw.models.Comment) -> bool:
        """Save comment to database."""
        try:
            current_time = int(datetime.now().timestamp())
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR IGNORE INTO comments (
                        id, subreddit, created_utc, author, score, body,
                        permalink, post_id, parent_id, is_submitter, processed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    comment.id,
                    str(comment.subreddit),
                    int(comment.created_utc),
                    str(comment.author) if comment.author else '[deleted]',
                    comment.compute_interaction_score,
                    comment.body,
                    comment.permalink,
                    comment.submission.id,
                    comment.parent_id,
                    1 if comment.is_submitter else 0,
                    current_time
                ))
                conn.commit()
                return True
        except Exception as e:
            self.logger.error(f"Error saving comment {comment.id} from r/{comment.subreddit}: {e}")
            return False

    def get_statistics(self) -> dict:
        """Get detailed statistics for all subreddits."""
        table = "posts" if self.content_type == "posts" else "comments"
        item_type = self.content_type[:-1]

        with sqlite3.connect(self.db_path) as conn:
            # Get per-subreddit statistics
            cursor = conn.execute(f"""
                SELECT 
                    subreddit,
                    COUNT(*) as item_count,
                    MIN(datetime(created_utc, 'unixepoch')) as first_item,
                    MAX(datetime(created_utc, 'unixepoch')) as last_item,
                    AVG(score) as avg_score
                FROM {table}
                GROUP BY subreddit
            """)

            stats = {}
            for row in cursor:
                stats[row[0]] = {
                    'item_count': row[1],
                    f'first_{item_type}': row[2],
                    f'last_{item_type}': row[3],
                    'avg_score': round(row[4], 2)
                }

            # Get recent activity
            cursor = conn.execute(f"""
                SELECT COUNT(*) as total_items 
                FROM {table}
            """)

            row = cursor.fetchone()  # Get the single row with total count
            if row:
                stats['total'][f'{self.content_type}_total'] = row[0]
            return stats


def main(content_type):
    # content_type = 'posts'
    # Example usage
    subreddits = load_subreddit_list()
    db_path = get_reddit_db_path()
    # Crawl comments
    crawler = RedditCrawler(subreddits, content_type, db_path)
    print("\nInitial comments statistics:")
    if content_type == "posts":
        batch_size = 100
    else:
        batch_size = 1000
    crawler.crawl(batch_size)


if __name__ == "__main__":
    fire.Fire(main)
