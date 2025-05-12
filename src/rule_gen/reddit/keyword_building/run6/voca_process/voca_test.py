import json
import os

from transformers import BertTokenizer

from chair.list_lib import list_equal
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_rp_path


def main():
    subreddit_list = get_split_subreddit_list("train")
    base_model = 'bert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(base_model)
    tokenize_fn = tokenizer.basic_tokenizer.tokenize
    for sb in subreddit_list:
        print(sb)
        save_path = get_rp_path( "s9_ngram_93", f"{sb}.json")
        if not os.path.exists(save_path):
            continue

        items = json.load(open(save_path))
        for res in items:
            for k in res["strong_sub_texts"]:
                sub_text: str = k["sub_text"]
                out1 = tokenizer.tokenize(sub_text)

                tokens = tokenize_fn(sub_text)
                out2 = tokenizer.tokenize(" ".join(tokens))

                if not list_equal(out1, out2):
                    print("sub_text", sub_text)
                    print(tokens)
                    print(out1)
                    print(out2)
                    print("-----")


def punkt():
    for cp in range(256):
        if (cp >= 33 and cp <= 47) or (cp >= 58 and cp <= 64) or (cp >= 91 and cp <= 96) or (cp >= 123 and cp <= 126):
            print(chr(cp), end=" ")


if __name__ == "__main__":
    punkt()
