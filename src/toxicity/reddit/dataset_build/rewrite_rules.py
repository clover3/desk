from toxicity.apis.open_ai_batch_requester import BatchChatGPTSender, BatchChatGPTLoader
from toxicity.cpath import output_root_path
from toxicity.reddit.path_helper import load_subreddit_list, load_reddit_rule
import os


def issue():
    instruction = "Rewrite the rule above. Keep all keywords, while removing repetition. Make the style concise. Goal is to compress it to reduce prompt length."
    sb_names = load_subreddit_list()
    batch_name = "rule_rewrite2"
    sender = BatchChatGPTSender(batch_name, "gpt-4o")
    for sb in sb_names:
        try:
            rules = load_reddit_rule(sb)
            all_text = ""
            for rule in rules:
                for role in ["summary", "detail"]:
                    all_text += rule[role] + " "
            prompt = f"{all_text}\n\n{instruction}"
            sender.add_request(sb, prompt)
        except FileNotFoundError as e:
            print(e)
    sender.submit_request()


def load():
    batch_name = "rule_rewrite2"
    sender = BatchChatGPTLoader(batch_name)
    sender.prepare_response()

    for k, v in sender.custom_id_to_response.items():
        sb = k
        rule_save_path = os.path.join(
            output_root_path, "reddit", "rules_para", f"{sb}.txt")
        open(rule_save_path, "w").write(v)


if __name__ == "__main__":
    # issue()
    load()
