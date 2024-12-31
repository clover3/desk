from rule_gen.reddit.build_rule.gpt_example_gen import build_prompt_from_true_false_items, build_rule_and_save
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    run_name = "chatgpt2"
    k = 50

    subreddit_list = get_split_subreddit_list("train")
    for sb in subreddit_list[1:3]:
        # load K true/false items
        prompt = build_prompt_from_true_false_items(sb, k)
        # print(prompt)
        build_rule_and_save(prompt, run_name, sb)


if __name__ == "__main__":
    main()
