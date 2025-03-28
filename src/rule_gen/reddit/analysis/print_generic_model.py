
import os
import sys

from rule_gen.reddit.path_helper import load_subreddit_list


import os
def find_files_without_keywords(keywords: list[str], dir_path: str) -> list[str]:
    result = []
    lowercase_keywords = [keyword.lower() for keyword in keywords]

    # Get list of all direct items in the directory
    try:
        items = os.listdir(dir_path)

        # Check each item
        for item in items:
            # Check if any keyword is in the item name
            if not any(keyword in item.lower() for keyword in lowercase_keywords):
                result.append(os.path.join(dir_path, item))

    except FileNotFoundError:
        print(f"Directory not found: {dir_path}")
    except PermissionError:
        print(f"Permission denied: {dir_path}")

    return result


def main():
    sb = load_subreddit_list()
    ret = find_files_without_keywords(sb, sys.argv[1])
    for t in ret:
        print(t)
    return NotImplemented



if __name__ == "__main__":
    main()