from collections import defaultdict

from desk_util.io_helper import read_csv, save_csv
from desk_util.path_helper import get_csv_dataset_path, get_label_path
from desk_util.runnable.run_eval import load_labels
from rule_gen.reddit.dataset_build3.build_dataset3 import load_deletion_rate
from rule_gen.reddit.path_helper import load_subreddit_list
import random


def load_del_rate_w_default():
    all_del_rate = 0.037
    del_rate_d: dict[str, float] = load_deletion_rate()


    return defaultdict(lambda: all_del_rate, del_rate_d)


def main():
    d = load_del_rate_w_default()
    print(min(d.values()))
    return
    subreddit_list = load_subreddit_list()
    for subreddit in subreddit_list:
        for split in ["train", "val", "test"]:
            dataset_name = f"{subreddit}_2_{split}"
            new_dataset_name = f"{subreddit}_2ub_{split}"

            data: list[tuple[str, str]] = read_csv(get_csv_dataset_path(dataset_name))
            labels: list[tuple[str, int]] = load_labels(dataset_name)

            pos_items = []
            neg_items = []

            for (_data_id, text), (_data_id2, label) in zip(data, labels):
                if label:
                    pos_items.append((_data_id, text))
                else:
                    neg_items.append((_data_id, text))

            del_rate = d[subreddit]
            n_pos = 50
            n_neg = int(n_pos / del_rate - n_pos)

            # Sample from positive and negative items
            sampled_pos = random.sample(pos_items, min(n_pos, len(pos_items)))
            sampled_neg = random.sample(neg_items, min(n_neg, len(neg_items)))

            # Combine samples and shuffle
            new_data = sampled_pos + sampled_neg
            random.shuffle(new_data)

            # Write to new dataset
            save_csv(new_data, get_csv_dataset_path(new_dataset_name))

            # Create labels for new dataset
            new_labels = [(data_id, 1 if (data_id, text) in pos_items else 0) for data_id, text in new_data]
            save_csv(new_labels, get_label_path(new_dataset_name))

            print(
                f"Created {new_dataset_name} with {len(sampled_pos)} positive and {len(sampled_neg)} negative examples")


if __name__ == "__main__":
    main()