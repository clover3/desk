import logging

import pandas as pd
from datasets import Dataset

LOG = logging.getLogger(__name__)


def load_dataset_from_csv(data_path):
    df = pd.read_csv(data_path,
                     na_filter=False, keep_default_na=False,
                     header=None, names=['text', 'label'], dtype={"text": str, 'label': int})
    for _, row in df.iterrows():
        if not isinstance(row['text'], str):
            print(row)
            raise ValueError
    return Dataset.from_pandas(df)
