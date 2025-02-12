from desk_util.open_ai import OpenAIChatClient
from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path, get_reddit_auto_prompt_path, \
    get_split_subreddit_list, get_reddit_train_data_path_ex


def build_prompt_from_k_true_false_items(sb, k):
    neg_items, pos_items = get_true_false_balanced(k, sb)
    return build_prompt_from_true_false_items(sb, pos_items, neg_items)


def get_true_false_balanced(k, sb):
    save_path = get_reddit_train_data_path(sb, "train")
    items = read_csv(save_path)
    pos_items = []
    neg_items = []
    for text, label in items:
        if label == "1" and len(pos_items) < k:
            pos_items.append(text)

        if label == "0" and len(neg_items) < k:
            neg_items.append(text)

        if len(pos_items) == k and len(neg_items) == k:
            break
    return neg_items, pos_items


def get_first_k(k, sb):
    save_path = get_reddit_train_data_path_ex("train_data2", sb, "train")
    items = read_csv(save_path)
    pos_items = []
    neg_items = []
    for text, label in items[:k]:
        if label == "1" and len(pos_items) < k:
            pos_items.append(text)

        if label == "0" and len(neg_items) < k:
            neg_items.append(text)

        if len(pos_items) == k and len(neg_items) == k:
            break
    return neg_items, pos_items


def build_prompt_from_true_false_items(sb, pos_items, neg_items):
    def wrap(text):
        return f"<text>{text}</text>"

    def combine_texts(texts):
        itr = map(wrap, texts)
        return "\n".join(itr)

    inst = (f"These are the examples of deleted posts and not deleted posts in subreddit {sb}. "
            f"Could you guess why they are deleted?")
    p_text = combine_texts(pos_items)
    n_text = combine_texts(neg_items)
    prompt = f"{inst} \n<deleted>\n {p_text} \n</deleted> \n<not deleted>\n {n_text} \n</not deleted>"
    prompt += "\n === Based on above, could you make list of rules?"
    prompt += f"Starts with: The following is a list of rules for deletions on the {sb} subreddit:"
    return prompt



def build_prompt_from_json_importance(sb, pos_items, neg_items):
    def wrap(text):
        return f"<text>{text}</text>"

    def combine_texts(texts):
        itr = map(wrap, texts)
        return "\n".join(itr)

    inst = (f"These are the examples of deleted posts and not deleted posts in subreddit {sb}. "
            f"Could you guess why they are deleted?")
    p_text = combine_texts(pos_items)
    n_text = combine_texts(neg_items)
    prompt = f"{inst} \n<deleted>\n {p_text} \n</deleted> \n<not deleted>\n {n_text} \n</not deleted>"
    prompt += "\n === Based on above, could you make list of rules?"
    prompt += f"Starts with: The following is a list of rules for deletions on the {sb} subreddit:"
    return prompt

def build_rule_and_save(prompt, run_name, sb):
    client = OpenAIChatClient()
    response2 = client.request(prompt)
    save_path = get_reddit_auto_prompt_path(run_name, sb)
    open(save_path, "w").write(response2)



def main():
    run_name = "chatgpt1"
    k = 10

    subreddit_list = get_split_subreddit_list("val")
    for sb in subreddit_list[1:]:
        # load K true/false items
        prompt = build_prompt_from_k_true_false_items(sb, k)
        # print(prompt)
        build_rule_and_save(prompt, run_name, sb)


if __name__ == "__main__":
    main()
