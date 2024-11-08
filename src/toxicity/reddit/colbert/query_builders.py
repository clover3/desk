import json

from toxicity.reddit.path_helper import get_reddit_rule_path, load_subreddit_list


def load_rule_text(role, sb):
    try:
        rule_save_path = get_reddit_rule_path(sb)
        rules = json.load(open(rule_save_path, "r"))
        if role == "both":
            rule_text = " ".join([r["summary"] + ". " + r["detail"] for r in rules])
        else:
            rule_text = " ".join([r[role] for r in rules])
    except FileNotFoundError:
        rule_text = f"Follow the rules that are appropriate for {sb} subreddit."
        print("Rule not found for {}. Replace to {}".format(sb, rule_text))
    return rule_text


def load_rule_d(role):
    sb_list = load_subreddit_list()
    return {sb: load_rule_text(role, sb) for sb in sb_list}


def get_sb_to_query(sb_strategy):
    if sb_strategy == "name":
        def sb_to_query(sb):
            return sb
    elif sb_strategy == "summary":
        rule_d = load_rule_d("summary")

        def sb_to_query(sb):
            return rule_d[sb]
    else:
        raise ValueError(sb_strategy)

    return sb_to_query