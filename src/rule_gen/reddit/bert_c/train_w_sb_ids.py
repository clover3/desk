from rule_gen.reddit.path_helper import load_subreddit_list


def load_sb_name_to_id_mapping() -> dict[str, int]:
    sb_names = load_subreddit_list()
    i = 1
    map_dict = {}
    for sb in sb_names:
        map_dict[sb] = i
        i += 1
    map_dict["_unknown_"] = 0
    return map_dict
