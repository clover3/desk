import random
from typing import List, Optional

from desk_util.io_helper import read_csv, save_csv
from rule_gen.reddit.dataset_build.build_train_mix3 import save_dataset
from rule_gen.reddit.path_helper import get_reddit_train_data_path_ex, get_split_subreddit_list


def load_and_sample_data(src_data_name, subreddit: str, split: str, max_samples: int) -> List[tuple[str, str, str]]:
    path = get_reddit_train_data_path_ex(src_data_name, subreddit, split)
    data = read_csv(path)
    random.shuffle(data)
    data = data[:max_samples]
    data = [(subreddit, text, label) for text, label in data]
    return data


def enum_sb_load_sample_shuffle_select(output_name: str, src_data_name: str,

                                       samples_per_subreddit: int, split, final_size: Optional[int]) -> None:
    subreddit_list = get_split_subreddit_list("train")
    all_data = []
    for subreddit in subreddit_list:
        data = load_and_sample_data(
            src_data_name,
            subreddit,
            split,
            samples_per_subreddit
        )
        all_data.extend(data)

    random.shuffle(all_data)
    if final_size is not None:
        all_data = all_data[:final_size]
    save_dataset(all_data, output_name, split)
    output_path = get_reddit_train_data_path_ex(src_data_name, output_name, split)
    save_csv(all_data, output_path)


def build_sampled_train_dataset(src_data_name, output_name: str, samples_per_subreddit: int, val_final_size: int) -> None:
    enum_sb_load_sample_shuffle_select(
        output_name,
        src_data_name,
        samples_per_subreddit,
        "train",
        None
    )
    enum_sb_load_sample_shuffle_select(
        output_name,
        src_data_name,
        samples_per_subreddit,
        "val",
        val_final_size
    )


if __name__ == "__main__":
    src_data_name = "train_data2"

    build_sampled_train_dataset(
        "train_data2",
        "train_comb3",
        200,
        1000)


# In N Out
