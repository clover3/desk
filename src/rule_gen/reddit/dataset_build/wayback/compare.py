from rule_gen.reddit.dataset_build.wayback.parse_rule import get_reddit_scratch_rule_path
from rule_gen.reddit.path_helper import load_subreddit_list, load_reddit_rule


def main():
    sb_names = load_subreddit_list()
    for sb in sb_names:
        try:
            p = get_reddit_scratch_rule_path(sb)
            rule_text = open(p, "r").read()
            print(">> Old")
            print(rule_text)
            rule_itmes = load_reddit_rule(sb)
            item = "\n".join([rule["summary"] + " " + rule["detail"] for rule in rule_itmes])
            print(">> New")
            print(item)
            print("====")
        except FileNotFoundError:
            pass


        # n_success += 1
        # try_parsing(soup)


if __name__ == "__main__":
    main()