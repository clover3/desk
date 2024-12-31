import random

from rule_gen.reddit.dataset_build.build_train_mix3 import load_and_sample_data, save_dataset
from rule_gen.reddit.path_helper import get_split_subreddit_list


def build_train(output_name: str, samples_per_subreddit: int) -> None:
    subreddit_list = get_split_subreddit_list("train")

    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            subreddit=subreddit,
            split="train",
            max_samples=samples_per_subreddit
        )
        data = [(subreddit, text, label) for text, label in data]
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
        data = [(subreddit, text, label) for text, label in data]
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
    # build_all("train_comb1", 1000, 1000)
    build_all("train_comb2", 200, 1000)
