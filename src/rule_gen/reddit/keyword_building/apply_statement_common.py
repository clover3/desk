import csv
import json

from chair.misc_lib import TimeEstimator
from desk_util.io_helper import read_csv
from desk_util.open_ai import OpenAIChatClient
from rule_gen.reddit.keyword_building.path_helper import get_keyword_statement_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def load_keyword_statement(sb) -> list[tuple[str, str]]:
    parsed_path = get_keyword_statement_path(sb)
    return json.load(open(parsed_path, "r"))


def form_question(statement, text):
    prompt = f"Is the given statement correct according to the text?\n"
    prompt += f"Answer Yes or No, as a single token.\n"
    prompt += f"<statement>{statement}</statement>\n"
    prompt += f"<text>{text}</text>"
    return prompt


def load_train_first_100(sb) -> list[tuple[str, str]]:
    data_name = "train_data2"
    p = get_reddit_train_data_path_ex(data_name, sb, "train")
    return read_csv(p)[:100]


def apply_statement(keyword_statement, res_save_path, texts):
    n_req = len(keyword_statement) * len(texts)
    client = OpenAIChatClient("gpt-4o")
    out_f = open(res_save_path, "w")
    csv_writer = csv.writer(out_f)
    ticker = TimeEstimator(n_req)
    for k_idx, ks in enumerate(keyword_statement):
        keyword, statement = ks
        for t_idx, text in enumerate(texts):
            prompt = form_question(statement, text)
            ret_text = client.request(prompt)
            ret = "yes" in ret_text.lower()
            csv_writer.writerow([k_idx, t_idx, ret])
            ticker.tick()


statement_gen_prompt_fmt = """
keyword: {}
With the keyword above, write a statement like:
* This text contains A.
* This text is A.
* This text is considered A.

Only output a single statement that best matches. 
"""
