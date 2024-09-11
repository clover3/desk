import torch
from torch.utils.data import DataLoader

from clf_util import BinaryDataset
from datasets import load_dataset


class ModerationDatasetAsBinary(BinaryDataset):
    def __init__(self):
        # There is no split
        self.data = load_dataset("mmathys/openai-moderation-api-evaluation")["train"]
        self.label_to_id = {"safe": 0, "unsafe": 1}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        converted_item = self.convert_dict(item)

        return {
            'id': str(idx),  # Use the index as a string ID
            'text': converted_item['prompt'],
            'label': self.label_to_id[converted_item['label']]
        }

    @staticmethod
    def convert_dict(input_dict):
        prompt = input_dict['prompt']
        label_keys = set(input_dict.keys()) - {'prompt'}
        is_unsafe = any(input_dict[key] == 1 for key in label_keys)
        label = "unsafe" if is_unsafe else "safe"
        return {"prompt": prompt, "label": label}


def create_data_loader(dataset, batch_size):
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_batch)


def collate_batch(batch):
    texts = [item['text'] for item in batch]
    labels = torch.stack([torch.tensor(item['label']) for item in batch])
    return {'text': texts, 'label': labels}