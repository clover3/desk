import pandas as pd
from datasets import Dataset

from rule_gen.reddit.path_helper import get_reddit_train_data_path


def load_dataset_from_csv(data_path):
    df = pd.read_csv(data_path,
                     na_filter=False, keep_default_na=False,
                     header=None, names=['text', 'label'], dtype={"text": str, 'label': int})
    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            print(row)
            raise ValueError
    return Dataset.from_pandas(df)


def main():
    print("start")
    subreddit = "AskReddit"
    load_dataset_from_csv(get_reddit_train_data_path(subreddit, "train"))
    print('end')


if __name__ == "__main__":
    main()