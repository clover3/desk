import os

from chair.misc_lib import make_parent_exists
from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import load_reddit_rule, get_split_subreddit_list


def get_generic_rule_save_path(dataset_name):
    file_name: str = f"{dataset_name}.txt"
    save_path: str = os.path.join(output_root_path, "reddit", "rule_processing",
                                  "generic", file_name)
    make_parent_exists(save_path)
    return save_path


def main():
    client = OpenAIChatClient("gpt-4o")
    split = "train"
    subreddit_list = get_split_subreddit_list(split)
    inst_fmt = ("These are the rules for the \"{}\" subreddit. "
                "Select the rules that can be applied to other subreddits with different topic."
                "Only output the rules, not justifications"
                )

    for sb in subreddit_list[21:]:
        try:
            rules = load_reddit_rule(sb)
            rule_text = " ".join([r["summary"] + ". " + r["detail"] for r in rules])
            inst = inst_fmt.format(sb)
            prompt = f"<rules>{rule_text}</rules> {inst}"
            response = client.request(prompt)
            print(f"<< === {sb} ===== >>")
            print(response)
            open(get_generic_rule_save_path(sb), "w").write(response)
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()