import math
import os
import pickle
from collections import Counter

from toxicity.cpath import output_root_path
from toxicity.reddit.path_helper import load_subreddit_list


def cosine_similarity(counter1: Counter, counter2: Counter) -> float:
    # Get the set of all unique terms
    all_terms = set(counter1.keys()) | set(counter2.keys())

    # Compute the dot product
    dot_product = sum(counter1.get(term, 0) * counter2.get(term, 0) for term in all_terms)

    # Compute the magnitudes
    magnitude1 = math.sqrt(sum(counter1.get(term, 0) ** 2 for term in all_terms))
    magnitude2 = math.sqrt(sum(counter2.get(term, 0) ** 2 for term in all_terms))

    # Compute cosine similarity
    if magnitude1 * magnitude2 == 0:
        return 0.0  # Handle the case where one or both vectors are zero
    else:
        return dot_product / (magnitude1 * magnitude2)


def main():
    sb_list = load_subreddit_list()

    tf_d = {}

    for sb in sb_list:
        save_path = os.path.join(output_root_path, "reddit", "tf", f"{sb}.pkl")
        tf = pickle.load(open(save_path, "rb"))
        tf_d[sb] = tf

    sim_arr: dict[str, list] = {sb: list() for sb in sb_list}
    for sb1 in sb_list:
        for sb2 in sb_list:
            print(sb1, sb2)
            if sb1 < sb2:
                score = cosine_similarity(tf_d[sb1], tf_d[sb2])
                sim_arr[sb1].append((sb2, score))
                sim_arr[sb2].append((sb1, score))

    save_path = os.path.join(output_root_path, "reddit", f"tf_sim.pkl")
    pickle.dump(sim_arr, open(save_path, "wb"))


def load_pickle_from(file_path: str):
    with open(file_path, "rb") as f:
        return pickle.load(f)

def show_sim():

    sim_file_path = os.path.join(output_root_path, "reddit", f"tf_sim.pkl")
    sim_data = load_pickle_from(sim_file_path)
    k = 5

    for subreddit, similarities in sim_data.items():
        print(f"\nTop {k} similarities for {subreddit}:")

        sorted_similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

        for i, (similar_subreddit, score) in enumerate(sorted_similarities[:k], 1):
            print(f"{i}. {similar_subreddit}: {score:.4f}")


def get_most_sim(src: str, avail: list[str]) -> str:
    sim_file_path = os.path.join(output_root_path, "reddit", f"tf_sim.pkl")
    sim_data = load_pickle_from(sim_file_path)

    if src not in sim_data:
        raise ValueError(f"Source subreddit '{src}' not found in similarity data")

    similarities = sim_data[src]

    available_similarities = [(sub, score) for sub, score in similarities if sub in avail]

    if not available_similarities:
        raise ValueError(f"No available subreddits found in similarity data for '{src}'")

    sorted_similarities = sorted(available_similarities, key=lambda x: x[1], reverse=True)

    return sorted_similarities[0][0]


if __name__ == "__main__":
    show_sim()
