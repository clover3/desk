import csv
import json
import os

import fire

from chair.list_lib import left
from chair.misc_lib import make_parent_exists, TimeEstimator
from desk_util.open_ai import OpenAIChatClient
from rule_gen.cpath import output_root_path
from desk_util.io_helper import read_csv
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


def load_keyword_statement(sb) -> list[str]:
    parsed_path = os.path.join(
        output_root_path, "reddit", "rule_processing", "keyword_statement", f"{sb}.json")
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


def main(sb):
    client = OpenAIChatClient("gpt-4o")
    keyword_statement = load_keyword_statement(sb)
    data = load_train_first_100(sb)
    texts = left(data)
    res_save_path = os.path.join(output_root_path, "reddit",
                                 "rule_processing", "k_to_text_100", f"{sb}.csv")
    make_parent_exists(res_save_path)
    n_req = len(keyword_statement) * len(texts)
    ticker = TimeEstimator(n_req)
    out_f = open(res_save_path, "w")
    csv_writer = csv.writer(out_f)
    for k_idx, ks in enumerate(keyword_statement):
        keyword, statement = ks
        for t_idx, text in enumerate(texts):
            prompt=  form_question(statement, text)
            ret_text = client.request(prompt)
            ret = "yes" in ret_text.lower()
            csv_writer.writerow([k_idx, t_idx, ret])
            ticker.tick()


if __name__ == "__main__":
    fire.Fire(main)