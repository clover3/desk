from rule_gen.reddit.classifier_loader.load_by_name import get_instruction_by_name
from rule_gen.reddit.path_helper import get_split_subreddit_list


def main():
    subreddit_list = get_split_subreddit_list("val")
    for sb in subreddit_list:
        try:
            inst = get_instruction_by_name(f"api_{sb}_both", "unsafe")
            print(sb, len(inst))
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()