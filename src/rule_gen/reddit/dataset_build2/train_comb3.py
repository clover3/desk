import random
from typing import List

from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.dataset_build.build_train_mix3 import save_dataset
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list


def load_and_sample_data(subreddit: str, split: str, max_samples: int) -> List[tuple[str, str, str]]:
    data_name = "train_data2"
    path = get_reddit_train_data_path_ex(data_name, subreddit, split)
    data = read_csv(path)
    random.shuffle(data)
    data = data[:max_samples]
    data = [(subreddit, text, label) for text, label in data]
    return data


def build_train(output_name: str, samples_per_subreddit: int) -> None:
    subreddit_list = get_split_subreddit_list("train")
    split = "train"
    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            subreddit=subreddit,
            split=split,
            max_samples=samples_per_subreddit
        )
        all_data.extend(data)

    random.shuffle(all_data)
    save_dataset(all_data, output_name, "train")
    output_path = get_reddit_train_data_path_ex("train_data2", output_name, split)
    save_csv(all_data, output_path)


def build_val(output_name: str, samples_per_subreddit: int, final_size: int) -> None:
    subreddit_list = get_split_subreddit_list("train")
    split = "val"
    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            subreddit=subreddit,
            split=split,
            max_samples=samples_per_subreddit
        )
        all_data.extend(data)

    random.shuffle(all_data)
    all_data = all_data[:final_size]
    save_dataset(all_data, output_name, "val")
    output_path = get_reddit_train_data_path_ex("train_data2", output_name, split)
    save_csv(all_data, output_path)



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
    build_all("train_comb3", 200, 1000)
