import os

from bs4 import BeautifulSoup

from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_subreddit_list
import re


def get_reddit_archive_save_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "wayback", f"{sb}.html")
    return rule_save_path


def get_reddit_scratch_rule_path(sb):
    rule_save_path = os.path.join(
        output_root_path, "reddit", "wayback", "rule_scratch", f"{sb}.txt")
    return rule_save_path


def parse_by_ol_ul(soup):
    maybe_rule = None
    pattern = re.compile('Rules', re.IGNORECASE)
    regex_matches = soup.find_all(string=pattern)
    for m in regex_matches:
        rule_span = m
        if "post" in rule_span.lower():
            print("Warning", rule_span)
            raise
        pointer = rule_span

        last_pointer = pointer

        while len(pointer.text) < len(rule_span.text) + 20:
            last_pointer = pointer
            pointer = pointer.parent
        # print("last_pointer", last_pointer)
        for list_like in pointer.find_all("ul"):
            text = list_like.text.lower()
            if "no " in text or "do not" in text:
                maybe_rule = list_like.text
                break
        for list_like in pointer.find_all("ol"):
            text = list_like.text.lower()
            if "no " in text or "do not" in text:
                maybe_rule = list_like.text
                break

    # print('maybe_rule', maybe_rule)
    return maybe_rule


def try_parsing(soup):
    maybe_rule = None
    pattern = re.compile('Rules', re.IGNORECASE)
    regex_matches = soup.find_all(string=pattern)
    for m in regex_matches:
        rule_span = m
        pointer = rule_span
        last_pointer = pointer

        while len(pointer.text) < len(rule_span.text) + 20:
            last_pointer = pointer
            pointer = pointer.parent

        print("Pointer ===")
        print(pointer)


    return maybe_rule



def main():
    sb_names = load_subreddit_list()
    n_success = 0
    for sb in sb_names:
        try:
            html_path = get_reddit_archive_save_path(sb)
            html_doc = open(html_path, "r", encoding="utf-8")
            soup = BeautifulSoup(html_doc, 'html.parser')
            maybe_rule = parse_by_ol_ul(soup)
            if maybe_rule is not None:
                p = get_reddit_scratch_rule_path(sb)
                # open(p, "w").write(maybe_rule)
                n_success += 1
                # try_parsing(soup)
                # break
        except Exception as e:
            print(e)
            # print(soup)


if __name__ == "__main__":
    main()
