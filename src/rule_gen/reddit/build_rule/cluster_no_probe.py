import json
import os

from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from rule_gen.reddit.build_rule.gpt_example_gen import build_prompt_from_json_importance
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path


def build_prompt_from_json_importance(j):
    inst = (f"Here are five texts in json format. "
            f"what are common in these texts? Answer as a list of statements.?")

    prompt = inst + " \n" + str(j)
    return prompt


def main():
    run_name = "cluster"
    # subreddit_list = get_split_subreddit_list("val")
    subreddit_list = get_split_subreddit_list("train")
    client = OpenAIChatClient()
    for sb in subreddit_list:
        print("Building prompt for", sb, "...")
        j_save_path = os.path.join(output_root_path, "reddit", "clusters_important", f"bert2_{sb}.json")
        j = json.load(open(j_save_path, "r"))
        response_list = []
        for per_cluster in j:
            text_list = [item["text"] for item in per_cluster]
            prompt = build_prompt_from_json_importance(text_list)
            response = client.request(prompt)
            response_list.append(response)

        save_path = get_reddit_auto_prompt_path(run_name, sb)
        json.dump(response_list, open(save_path, "w"))


if __name__ == "__main__":
    main()
