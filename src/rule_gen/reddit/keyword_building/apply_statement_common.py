import csv
import os

from chair.misc_lib import TimeEstimator
from desk_util.io_helper import read_csv
from desk_util.open_ai import OpenAIChatClient
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex


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
    # Initialize tracking variables
    completed_pairs = set()

    # Read existing results if file exists
    if os.path.exists(res_save_path):
        with open(res_save_path, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                k_idx, t_idx = int(row[0]), int(row[1])
                completed_pairs.add((k_idx, t_idx))

    print("{} record exists".format(len(completed_pairs)))
    # Calculate remaining work
    total_pairs = len(keyword_statement) * len(texts)
    remaining_pairs = total_pairs - len(completed_pairs)

    client = OpenAIChatClient("gpt-4o")
    ticker = TimeEstimator(remaining_pairs)

    # Open file in append mode to preserve existing results
    with open(res_save_path, "a", newline='') as out_f:
        csv_writer = csv.writer(out_f)

        for k_idx, ks in enumerate(keyword_statement):
            keyword, statement = ks
            for t_idx, text in enumerate(texts):
                if (k_idx, t_idx) in completed_pairs:
                    continue

                prompt = form_question(statement, text)
                ret_text = client.request(prompt)
                ret = "yes" in ret_text.lower()
                csv_writer.writerow([k_idx, t_idx, ret])
                ticker.tick()

                out_f.flush()

statement_gen_prompt_fmt = """
keyword: {}
With the keyword above, write a statement like:
* This text contains A.
* This text is A.
* This text is considered A.

Only output a single statement that best matches. 
"""
