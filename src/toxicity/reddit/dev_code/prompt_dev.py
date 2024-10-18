import json

from toxicity.llama_helper.lf_client import LLMClient
from toxicity.reddit.path_helper import get_reddit_rule_path


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
    rule_text = " ".join([r['summary'] for r in rules])
    rule_text = ""

    inst_core = ("The following two texts are deleted from 'askscience' subreddit. "
                 # "The above is list of existing rules."
                 "Could you guess why this text is deleted? "
                 "Could make a new rule statement corresponds to that reason?\n"
                 )

    system = f"{rule_text}\n===\n{inst_core}"
    prompt = f"<text>{text}</text>"
    ret = client.ask(prompt, system)
    print(ret)


if __name__ == "__main__":
    main()

