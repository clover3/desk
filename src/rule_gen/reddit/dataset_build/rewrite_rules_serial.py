import os

from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule


def main():
    instruction = ("Rewrite the rule above. Keep all keywords, while removing repetition. "
                   "Make the style concise. Goal is to compress it to reduce prompt length.")
    sb_names = load_subreddit_list()
    client = OpenAIChatClient("gpt-4o")
    for sb in sb_names:
        try:
            rules = load_reddit_rule(sb)
            all_text = ""
            for rule in rules:
                for role in ["summary", "detail"]:
                    all_text += rule[role] + " "
            prompt = f"{all_text}\n\n{instruction}"
            v = client.request(prompt)
            print(v)
            rule_save_path = os.path.join(
                output_root_path, "reddit", "rules_para", f"{sb}.txt")
            open(rule_save_path, "w").write(v)

        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
