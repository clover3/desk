import json

from toxicity.llama_helper.lf_client import LLMClient
from toxicity.reddit.path_helper import get_reddit_rule_path
from toxicity.reddit.prompt_opt.enum_wrong import enum_wrong, enum_FN
from toxicity.reddit.prompt_opt.optimizer import parse_tagged_text


def main():
    client = LLMClient()
    text = """
    Thank you for your submission! Unfortunately, your submission has been automatically removed because it contains the phrase "**ELI5**", so it is possible you are looking for /r/explainlikeimfive. If you would like scientific answers, you can repost your question to /r/AskScience though! Experts will always simplify and explain, so that even difficult concepts are easy to understand. Thanks for understanding. :)

*I am a bot, and this action was performed automatically. Please [contact the moderators of this subreddit](/message/compose/?to=/r/askscience) if you have any questions or concerns.*
    """
#     text1 = """
#     Basically the answer is no, nothing like The Fertile Crescent, or Egypt, or the sheer number of developing civilizations east of the Atlantic Ocean prior to the 1000s BC or so
#
# There were sporadic and far less advanced civilizations with perhaps some kind of language system (usually limited to elites) and maybe some basic writing and accounting (literacy was limited to *very* few nonetheless) in the western world, but it was nothing like what is seen in the Middle East at the time. And saying "at the time" is stretching the truth, because I'm talking about the ancient civilizations of the Mayans, Incas, and Aztecs, and the like, who came significantly (millennia) after the Mesopotamians.
#
# Bottom line, there were simply no rich and lucrative civilizations at the same time as those early in the Middle East, and those that could vaguely be considered so we're less able to flourish because the distance between themselves and nearest other civilizations was just too far.
#
# Believe it or not, the reason for this discrepancy across continents, according to scholarly consensus, is the lack of domesticable animals in the Western Hemisphere. If you're interested further, I highly suggest the famous book *Guns, Germs, and Steel* by Jared Diamond. You'll learn more than you bargained for, and you'll enjoy it all along."""
    text = "Who knows, it may continue. For all we know, the big bang might not have just been one point in nothingness that started to expand. It might have been one if infinitely many points, but our universe was contained within it, and is expanding at the speed of light. The point right next to our big bang is also moving away from our point, staying outside rhe boundary, also expanding."
    sb = "askscience"
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text = " ".join([r['detail'] for r in rules])

    inst_core = ("The following post is deleted from 'askscience' subreddit, but it is not covered by existing rules "
                 # "The above is list of existing rules."
                 "Could you guess why this text is deleted? "
                 # "Could make a new rule statement corresponds to that reason?\n"
                 )

    system = f"<existing rules>{rule_text}</existing rules>\n===\n{inst_core}"
    prompt = f"<text>{text}</text>"
    ret = client.ask(prompt, system)
    print(ret)


def get_new_classifier(sb, role, new_rule: list[str], pos_keyword="unsafe"):
    assert role in ["summary", "detail"]
    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text_list = [r[role] for r in rules]
    rule_text_list.extend(new_rule)
    rule_text = " ".join(rule_text_list)
    inst_summary = "The above rule describes prohibited contents. Classify if the following text is prohibited. "
    inst_summary += f"If prohibited, output '{pos_keyword}' as a first token. If not, output 'safe'"
    instruction = rule_text + "\n " + inst_summary

    client = LLMClient(max_prompt_len=5000)
    pos_keyword = "unsafe"

    def predict(text):
        ret_text = client.ask(text, instruction)
        print("Ret_text", ret_text)
        pred = pos_keyword in ret_text.lower()
        ret = int(pred)
        return ret, 0
    return predict


def one_rule_at_a_time():
    client = LLMClient()
    sb = "askscience"
    run_name = f"api_{sb}_detail"
    entries = list(enum_FN(f"{sb}_val_100", run_name))

    entries = [e for e in entries if "thank you for submitting to" not in e["text"]]
    text_list = [e["text"] for e in entries[1:2]]
    prompt = " ".join([f"<text>{t}</text>" for t in text_list])

    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    print(prompt)
    rule_text_list = [r['summary'] + ": " + r['detail'] for r in rules]
    for rule_text in rule_text_list:
        pos_keyword = "yes"
        system = f"<rule>{rule_text}</rule>\n===\nDoes the following text violates the given rule?\n"
        system += f"Output either '{pos_keyword}' or 'no' as the first token. "
        ret = client.ask(system + prompt)
        print(rule_text)
        print(ret)


def run_from_wrong():
    client = LLMClient()
    sb = "askscience"
    sb = "NeutralPolitics"
    run_name = f"api_{sb}_detail"
    entries = list(enum_FN(f"{sb}_val_100", run_name))
    entries = [e for e in entries if "thank you for submitting to" not in e["text"]]
    text_list = [e["text"] for e in entries[1:2]]
    prompt = " ".join([f"<text>{t}</text>" for t in text_list])

    rule_save_path = get_reddit_rule_path(sb)
    rules = json.load(open(rule_save_path, "r"))
    rule_text = " ".join([r['detail'] for r in rules])

    start_tag = "<rule>"
    end_tag = "</rule>"
    inst_core = ("The following post is deleted from 'askscience' subreddit, but it is not covered by existing rules "
                 "The above is list of existing rules."
                 "Could you guess why this text is deleted? "
                 "Could make a new rule statement corresponds to that reason?\n"
                 f"Wrap the rule with {start_tag} and {end_tag}."
                 )

    system = f"<rules>{rule_text}</rules>\n===\n{inst_core}\n"
    ret = client.ask(system + prompt)
    new_rule = parse_tagged_text(ret, start_tag, end_tag)
    print(prompt)
    print(ret)

    print("rule:", new_rule)
    predict = get_new_classifier(sb, "detail", new_rule)
    print(predict(text_list[0]))


if __name__ == "__main__":
    run_from_wrong()
