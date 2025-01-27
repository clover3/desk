import json
from json import JSONDecodeError

from chair.misc_lib import make_parent_exists
from rule_gen.reddit.keyword_building.keyword_extractor import parse_openai_json
from rule_gen.reddit.keyword_building.path_helper import get_keyword_req_response_path, get_parsed_keyword_path
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule2


def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        print(sb)
        try:
            raw_path = get_keyword_req_response_path(sb)
            data = json.load(open(raw_path, "r"))
            rules = load_reddit_rule2(sb)
            keywords = []
            for i, response in enumerate(data):
                ret = parse_openai_json(response)
                if not ret:
                    print("warning nothing parsed", response)
                    print("Corresponding rule: ", rules[i])

                keywords.extend(ret)

            parsed_path = get_parsed_keyword_path(sb)
            make_parent_exists(parsed_path)
            json.dump(keywords, open(parsed_path, "w"))
        except FileNotFoundError as e:
            print(e)
        except JSONDecodeError:
            print("JSONDecodeError")
            print(response)


if __name__ == "__main__":
    main()