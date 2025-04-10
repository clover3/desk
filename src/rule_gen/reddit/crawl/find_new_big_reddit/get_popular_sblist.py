import requests
import json
import time

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
import os

def crawl_subreddits(base_url, start_page=1, end_page=10, limit=10, category="sfw-largest", nsfw=False):
    for page in range(start_page, end_page + 1):
        url = f"{base_url}?type=largest&category={category}&nsfw={str(nsfw).lower()}&page={page}&limit={limit}&query=&filter="

        try:
            print(f"Crawling page {page}...")
            response = requests.get(url)

            # Check if the request was successful
            if response.status_code == 200:
                data = response.json()
                yield from data
                print(f"Successfully crawled page {page}, got {len(data)} items")
            else:
                print(f"Failed to retrieve data from page {page}. Status code: {response.status_code}")

            # Add a small delay to avoid hitting rate limits
            time.sleep(1)

        except Exception as e:
            print(f"Error crawling page {page}: {str(e)}")



def read_existing_jsonl(filename):
    """
    Read existing data from a JSONL file.

    Args:
        filename (str): Name of the input file

    Returns:
        tuple: (list of existing data, set of existing subreddit names)
    """
    existing_data = []
    existing_names = set()

    try:
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    existing_data.append(item)
                    if 'name' in item:
                        existing_names.add(item['name'])
                except json.JSONDecodeError:
                    continue

        print(f"Read {len(existing_data)} existing items from {filename}")
    except FileNotFoundError:
        print(f"No existing file found at {filename}. Will create a new one.")

    return existing_data, existing_names


def save_to_jsonl(data, filename, append=False):
    """
    Save data to a JSONL file.

    Args:
        data (list): List of dictionaries to save
        filename (str): Name of the output file
        append (bool): Whether to append to existing file
    """

    print(f"Successfully saved {len(data)} items to {filename}")


if __name__ == "__main__":
    # Base URL for the API
    base_url = "https://subranking.com/load_data"

    # Configuration
    limit = 1000  # Items per page
    n_page = 100

    # Choose your category and NSFW settings
    # For SFW subreddits
    category = "sfw-largest"
    nsfw = False
    file_name = "sfw_subreddits.jsonl"
    output_file = os.path.join(output_root_path, "reddit", "popular_list", file_name)
    make_parent_exists(output_file)

    # Uncomment the following for NSFW subreddits
    # category = "nsfw-largest"
    # nsfw = True
    # output_file = "nsfw_subreddits.jsonl"

    # Read existing data if available
    existing_data, existing_names = read_existing_jsonl(output_file)

    # Determine the next page to start from
    # If we have existing data, calculate the page we should start from
    if existing_data:
        # Assuming data is crawled in order and each page has exactly 'limit' items
        start_page = (len(existing_data) // limit) + 1
    else:
        start_page = 1

    end_page = start_page + n_page  # Crawl 10 more pages by default

    print(f"Starting crawl from page {start_page} to page {end_page}")

    # Crawl the new data
    sb_itr = crawl_subreddits(base_url, start_page, end_page, limit, category, nsfw)
    mode = 'a' if bool(existing_data) else 'w'

    with open(output_file, mode, encoding='utf-8') as f:
        for item in sb_itr:
            f.write(json.dumps(item) + '\n')
            f.flush()