import random
from typing import List

from toxicity.io_helper import read_csv, save_csv
from toxicity.reddit.path_helper import get_reddit_train_data_path, get_split_subreddit_list


def load_and_sample_data(subreddit: str, split: str, max_samples: int) -> List:
    path = get_reddit_train_data_path(subreddit, split)
    data = read_csv(path)
    random.shuffle(data)
    return data[:max_samples]


def save_dataset(data: List, output_name: str, split: str) -> None:
    output_path = get_reddit_train_data_path(output_name, split)
    save_csv(data, output_path)


def build_train(output_name: str, samples_per_subreddit: int) -> None:
    subreddit_list = get_split_subreddit_list("train")

    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            subreddit=subreddit,
            split="train",
            max_samples=samples_per_subreddit
        )
        all_data.extend(data)

    random.shuffle(all_data)
    save_dataset(all_data, output_name, "train")


def build_val(output_name: str, samples_per_subreddit: int, final_size: int) -> None:
    subreddit_list = get_split_subreddit_list("train")

    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            subreddit=subreddit,
            split="val",
            max_samples=samples_per_subreddit
        )
        all_data.extend(data)

    random.shuffle(all_data)
    all_data = all_data[:final_size]
    save_dataset(all_data, output_name, "val")


def build_all(output_name: str, samples_per_subreddit: int, val_final_size: int) -> None:
    build_train(
        output_name=output_name,
        samples_per_subreddit=samples_per_subreddit
    )
    build_val(
        output_name=output_name,
        samples_per_subreddit=samples_per_subreddit,
        final_size=val_final_size
    )


if __name__ == "__main__":
    build_all("train_mix3", 1000, 1000)
