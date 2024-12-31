from transformers import BertTokenizer

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


def main2():
    sb_names = load_subreddit_list()
    print("Paraphrased length")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    over500 = 0
    for sb in sb_names:
        try:
            rul_text = load_reddit_rule_para(sb)
            tokens = tokenizer.tokenize(rul_text)
            if len(tokens) > 500:
                over500 += 1
            print(sb, len(tokens))
        except FileNotFoundError as e:
            print(e)
    print("over500: ", over500)


if __name__ == "__main__":
    main2()
