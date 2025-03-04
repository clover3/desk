import fire
from transformers import BertTokenizer, AutoTokenizer

from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def get_tokens(full_text, sub_text_list, tokenizer, max_len=512):
    text_sp_rev = " ".join(full_text.split())
    sub_text_embeddings = []

    # Get full text embeddings
    encoded = tokenizer(
        text_sp_rev,
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=max_len
    )
    print(encoded["input_ids"])
    print(tokenizer.convert_ids_to_tokens(encoded["input_ids"][0]))
    for sub_text in sub_text_list:
        # Get the sub-text for this window
        # Get token positions for this sub-text in the original text
        sub_text_start_char = text_sp_rev.find(sub_text)
        sub_text_end_char = sub_text_start_char + len(sub_text) - 1

        st = encoded.char_to_token(0, sub_text_start_char)
        ed = encoded.char_to_token(0, sub_text_end_char) + 1
        print(sub_text, st, ed)

    return sub_text_embeddings


def main(sb= "TwoXChromosomes"):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    data_name = "train_data2"
    train_dataset = load_dataset_from_csv(get_reddit_train_data_path_ex(
        data_name, sb, "train"))
    for example in train_dataset:
        text = example['text']
        sp_tokens = text.split()
        sub_text_list = []
        for window_size in range(2, 4):
            for i in [0, 4, 8]:
                if i+window_size < len(sp_tokens):
                    sub_text = " ".join(sp_tokens[i: i+window_size])
                    sub_text_list.append(sub_text)
        get_tokens(text, sub_text_list, tokenizer)

if __name__ == "__main__":
    fire.Fire(main)