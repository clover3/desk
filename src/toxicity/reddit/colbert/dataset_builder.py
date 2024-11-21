import pandas as pd
from datasets import Dataset as HFDataset

from toxicity.reddit.train_common import DatasetLoader


class TwoColumnDatasetLoader(DatasetLoader):
    def __init__(self, constant_query="Please classify this text"):
        self.constant_query = constant_query

    def _get_query(self):
        return self.constant_query

    def get(self, data_path, max_samples=None):
        df = pd.read_csv(
            data_path,
            na_filter=False,
            keep_default_na=False,
            header=None,
            names=['text', 'label'],
            dtype={"text": str, 'label': int}
        )
        if max_samples is not None:
            df = df.head(max_samples)
        query = self._get_query()
        triplet_df = pd.DataFrame({
            'query': [query] * len(df),
            'document': df['text'],
            'label': df['label']
        })
        return HFDataset.from_pandas(triplet_df)

    def _create_triplet_df(self, df):
        query = self._get_query()
        return pd.DataFrame({
            'query': [query] * len(df),
            'document': df['text'],
            'label': df['label']
        })


class ThreeColumnDatasetLoader(DatasetLoader):
    def __init__(self, sb_to_query):
        self.sb_to_query = sb_to_query

    def get(self, data_path, max_samples=None):
        df = pd.read_csv(
            data_path,
            na_filter=False,
            keep_default_na=False,
            header=None,
            names=['sb', 'text', 'label'],
            dtype={"sb": str, "text": str, 'label': int}
        )
        if max_samples is not None:
            df = df.head(max_samples)

        triplet_df = pd.DataFrame({
            'query': df["sb"].apply(self.sb_to_query),  # Apply custom function here
            'document': df['text'],
            'label': df['label']
        })
        return HFDataset.from_pandas(triplet_df)
