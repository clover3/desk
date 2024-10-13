import json

from transformers import BertTokenizer

from toxicity.reddit.path_helper import load_subreddit_list, get_reddit_rule_path


def main():
    sb_names = load_subreddit_list()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for sb in sb_names:
        try:
            rule_save_path = get_reddit_rule_path(sb)
            rules = json.load(open(rule_save_path, "r"))
            for role in ["summary", "detail"]:
                all_summary = " ".join([rule[role] for rule in rules])
                tokens = tokenizer.tokenize(all_summary)
                print(sb, role, len(tokens))
        except FileNotFoundError as e:
            print(e)


if __name__ == "__main__":
    main()
