

def get_pattern_instruction(sb, patterns):
    inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
    inst += "Here are common patterns that are deleted: \n ["
    pattern_str = ",\n".join(["{} ".format(p) for p in patterns])
    inst += pattern_str + " ]"
    inst += f"\n    Answer Yes or No, as a single token.\n"
    return inst


def get_pattern_instruction2(sb, patterns):
    inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
    inst += "Here are common patterns that are deleted: \n"\
            "<Patterns>\n"
    pattern_str = "\n".join(["<pattern>{}</pattern> ".format(p) for p in patterns])
    inst += pattern_str + "\n</Patterns>"
    inst += "\nNote that there could be moderated texts which are not captured by these patterns."
    inst += f"\n    Answer Yes or No, as a single token.\n"
    return inst


def get_pattern_instruction_w_prepost(sb, prefix, postfix, patterns):
    inst_pre = prefix.format(sb)
    pattern_str = "\n".join(["<pattern>{}</pattern> ".format(p) for p in patterns])
    pattern_str = "<Patterns>\n" + pattern_str + "\n</Patterns>"
    inst_post = postfix.format(sb)
    return f"{inst_pre}\n\n{pattern_str}\n\n{inst_post}"

