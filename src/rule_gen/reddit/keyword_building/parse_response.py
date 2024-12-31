import json
import os
from json import JSONDecodeError

from chair.misc_lib import make_parent_exists
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule2



def parse_openai_json(s):
    lines = s.split("\n")
    if lines[0] == "```json" and lines[-1] == "```":
        raw_j = "\n".join(lines[1:-1])
        return json.loads(raw_j)
    if lines[2] == "```json" and lines[-1] == "```":
        raw_j = "\n".join(lines[3:-1])
        return json.loads(raw_j)
    elif lines[0] == "[" and lines[-1] == "]":
        raw_j = s
        return json.loads(raw_j)
    elif lines[0] == "{" and lines[-1] == "}":
        raw_j = s
        j = json.loads(raw_j)
        key = next(iter(j.keys()))
        return j[key]
    else:
        print("Parse failed", lines)
        raise ValueError()
    # s = s.strip()
    # if s.startswith("```json") and s.endswith("```"):
    #     st = len("```json")
    #     ed = -len("```")
    #     s = s[st:ed]
    #     return json.loads(s)
    # else:
    #     raise ValueError()

def main():
    sb_list = load_subreddit_list()
    for sb in sb_list:
        print(sb)
        try:
            raw_path = os.path.join(
                output_root_path, "reddit", "rule_processing", "keyword_raw", f"{sb}.json")
            data = json.load(open(raw_path, "r"))
            rules = load_reddit_rule2(sb)
            keywords = []
            for i, response in enumerate(data):
                ret = parse_openai_json(response)
                if not ret:
                    print("warning nothing parsed", response)
                    print("Corresponding rule: ", rules[i])

                keywords.extend(ret)

            parsed_path = os.path.join(
                output_root_path, "reddit", "rule_processing", "keyword", f"{sb}.json")
            make_parent_exists(parsed_path)
            json.dump(keywords, open(parsed_path, "w"))
        except FileNotFoundError as e:
            print(e)
        except JSONDecodeError:
            print("JSONDecodeError")
            print(response)


if __name__ == "__main__":
    main()