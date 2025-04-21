import numpy as np
from typing import Callable

from openai.types.beta import VectorStoreListParams

from rule_gen.reddit.classifier_loader.prompt_based import fetch_score_from_token_probs


def get_instruction(sb):
    inst = f"If the following text is posted in {sb} subreddit, will it be moderated (deleted)?\n"
    inst += f"Start a response with Yes or No, as a single token. If the response is 'Yes', explain why.\n"
    inst += "\n Text: "
    return inst


def get_instruction_from_run_name(run_name):
    tokens = run_name.split("_")
    p_name = tokens[1]
    sb = "_".join(tokens[2:])
    return get_instruction(sb)


def convert_entry(pred, raw_score):
    prob = np.exp(raw_score)
    if int(pred):
        score = prob
    else:
        # Higher score -> Lower confidence
        score = -prob

    return score



def get_llm_predictor(run_name) -> Callable[[str], tuple[int, float, str]]:
    from desk_util.open_ai import OpenAIChatClient
    client = OpenAIChatClient("gpt-4o")
    if run_name.startswith("chatgpt"):
        instruction = get_instruction_from_run_name(run_name)
        pos_keyword = "yes"
        def predict(text):
            prompt = instruction + text
            ret = client.request_with_probs(prompt)
            ret_text = ret["content"]
            token_probs = ret["token_probs"]
            score = fetch_score_from_token_probs(pos_keyword, token_probs)
            pred = pos_keyword.lower() in ret_text[:10].lower()
            label = int(pred)
            score = convert_entry(label, score)
            save_e = label, score, ret_text

            return save_e
    else:
        raise ValueError()

    return predict
