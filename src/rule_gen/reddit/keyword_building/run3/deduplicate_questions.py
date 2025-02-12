from json import JSONDecodeError

import numpy as np
import json
import os
from rule_gen.reddit.keyword_building.keyword_extractor import parse_openai_json
from rule_gen.cpath import output_root_path
from rule_gen.reddit.path_helper import get_split_subreddit_list, get_reddit_auto_prompt_path


def load_question_gen_response(run_name, sb):
    save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                             f"{run_name}_questions", f"bert2_{sb}.json")
    responses = json.load(open(save_path, "r", encoding="utf-8"))

    q_list: list[str] = []
    for r in responses:
        try:
            j: list[str] = parse_openai_json(r)
            q_list.extend(j)
        except JSONDecodeError as e:
            pass

    return q_list


from transformers import pipeline

# Load the LLM model

def get_finder(threshold=0.8):
    model = pipeline("feature-extraction", model="sentence-transformers/all-mpnet-base-v2")
    def find_near_duplicates(texts):
        embeddings = [np.mean(model(text), axis=1)[0] for text in texts]
        similarities = []

        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                similarity = np.dot(embeddings[i], embeddings[j]) / (
                            np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]))

                similarities.append((i, j, similarity))
        near_duplicates = [(pair[0], pair[1]) for pair in similarities if pair[2] > threshold]

        return near_duplicates
    return find_near_duplicates


def main():
    run_name = "cluster"
    subreddit_list = get_split_subreddit_list("train")
    finder = get_finder(0.95)
    for sb in subreddit_list:
        print(f"==== {sb} ====")
        try:
            q_list = load_question_gen_response(run_name, sb)
            ret = finder(q_list)
            skip_i = set()
            for i1, i2 in ret:
                skip_i.add(i1)

            new_q_list = []
            for i, text in enumerate(q_list):
                if i not in skip_i:
                    new_q_list.append(text)

            n_reduce = len(q_list) - len(new_q_list)
            print(n_reduce)
            save_path = os.path.join(output_root_path, "reddit", "rule_processing",
                                     f"{run_name}_questions_dedup", f"bert2_{sb}.json")
            json.dump(new_q_list, open(save_path, "w"))


        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
