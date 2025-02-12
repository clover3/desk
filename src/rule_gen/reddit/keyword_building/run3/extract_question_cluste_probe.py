import json
import os

from chair.misc_lib import make_parent_exists
from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path

question_gen_prompt_fmt = """
<text>
{}
</text>
===
Based on the text above, write questions like "is this text~ " or " does this text?". 
You can make multiple questions from single statement if the sentence is too long to fit in one question. 
Output results as json list of strings
"""


def main():
    run_name = "cluster_probe"
    subreddit_list = get_split_subreddit_list("train")

    client = OpenAIChatClient("gpt-4o")
    for sb in subreddit_list:
        print(f"==== {sb} ====")
        try:
            save_path = get_reddit_auto_prompt_path(run_name, sb)
            responses = json.load(open(save_path, "r"))
            q_save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                       f"{run_name}_questions", f"bert2_{sb}.json")
            make_parent_exists(q_save_path)
            output = []
            for r in responses:
                prompt = question_gen_prompt_fmt.format(r)
                ret_text = client.request(prompt)
                output.append((ret_text))
                print(ret_text)
            json.dump(output, open(q_save_path, "w"))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
