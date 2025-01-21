import json

from desk_util.open_ai import OpenAIChatClient
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path, get_reddit_rule_path


def main():
    subreddit_list = get_split_subreddit_list("val")
    for sb in subreddit_list[1:]:
        src_run_name = "chatgpt3"
        save_path = get_reddit_auto_prompt_path(src_run_name, sb)
        generated_rules = open(save_path, "r").read()
        rule_save_path = get_reddit_rule_path(sb)
        rules = json.load(open(rule_save_path, "r"))
        rule_text = " ".join([r["summary"] + ". " + r["detail"] for r in rules])

        prompt = "Select rules from <rule A> that has corresponding one in <rule B>.\n"
        prompt += "Generate a new set of rules based on result.\n"
        prompt += f"<rule A>{generated_rules} </rule A>\n"
        prompt += f"<rule B>{rule_text} </rule B>\n"
        client = OpenAIChatClient()
        response = client.request(prompt)
        print(sb)
        print(response)
        new_run_name = "chatgpt3_overlap"
        save_path = get_reddit_auto_prompt_path(new_run_name, sb)
        open(save_path, "w").write(response)

    return NotImplemented


if __name__ == "__main__":
    main()
