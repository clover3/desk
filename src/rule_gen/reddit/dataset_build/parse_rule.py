import json
import os

from bs4 import BeautifulSoup

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import data_root_path
from rule_gen.reddit.path_helper import load_subreddit_list, get_reddit_rule_path


def parse_for(sb):
    html_path = os.path.join(
        data_root_path, "reddit", "pages", f"{sb}.html")
    html_doc = open(html_path, "r", encoding="utf-8")
    soup = BeautifulSoup(html_doc, 'html.parser')
    detail = soup.find("details")
    maybe_root = detail.parent.parent

    output = []
    for item in maybe_root.find_all("details"):
        summary = item.find("summary")
        summary_text = summary.get_text(" ", strip=True)
        # print("summary:", summary.get_text(" ", strip=True))
        more_detail = summary.next_sibling.next_sibling
        detail_text = more_detail.get_text(" ", strip=True)
        # print("detail: ", more_detail.get_text(" ", strip=True))
        j = {"summary": summary_text, "detail": detail_text}
        output.append(j)
    return output


def main():
    sb_names = load_subreddit_list()

    for sb in sb_names:
        rule_save_path = get_reddit_rule_path(sb)
        make_parent_exists(rule_save_path)
        print("Sb", sb)
        try:
            rules_j = parse_for(sb)
            print(f"{len(rules_j)} rules")
            open(rule_save_path, "w").write(json.dumps(rules_j))
        except AttributeError as e:
            print(e)


if __name__ == "__main__":
    main()
