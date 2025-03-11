from transformers import BertTokenizer

from rule_gen.reddit.colbert.query_builders import load_rule_text
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule, load_reddit_rule_para


def main():
    sb_names = load_subreddit_list()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    for sb in sb_names:
        try:
            rules = load_reddit_rule(sb)
            for role in ["summary", "detail"]:
                all_summary = " ".join([rule[role] for rule in rules])
                tokens = tokenizer.tokenize(all_summary)
                print(sb, role, len(tokens))
        except FileNotFoundError as e:
            print(e)


def show_para_rule_len():
    sb_names = load_subreddit_list()
    print("Paraphrased length")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    over500 = 0
    for sb in sb_names:
        try:
            rul_text = load_reddit_rule_para(sb)
            orig = load_rule_text("both", sb)
            before_len = len(tokenizer.tokenize(orig))
            after_len = len(tokenizer.tokenize(rul_text))
            if after_len > 500:
                over500 += 1
            print(sb, before_len, after_len)
        except FileNotFoundError as e:
            print(e)
    print("over500: ", over500)


if __name__ == "__main__":
    show_para_rule_len()
