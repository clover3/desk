from desk_util.open_ai import OpenAIChatClient
from rule_gen.reddit.build_rule.gpt_example_gen import get_first_k, build_prompt_from_true_false_items
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path


def main():
    run_name = "chatgpt3"
    k = 10
    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list[:1]:
        neg_items, pos_items = get_first_k(k, sb)
        prompt = build_prompt_from_true_false_items(sb, pos_items, neg_items)
        client = OpenAIChatClient()
        response = client.request(prompt)
        save_path = get_reddit_auto_prompt_path(run_name, sb)
        open(save_path, "w").write(response)


if __name__ == "__main__":
    main()
