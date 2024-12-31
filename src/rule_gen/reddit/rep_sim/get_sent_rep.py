import faiss
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from transformers import BertTokenizer, BertModel

from chair.list_lib import left, right
from desk_util.io_helper import read_csv
from desk_util.path_helper import get_model_save_path
from rule_gen.reddit.path_helper import get_reddit_train_data_path


class TextDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(self.texts[idx], padding='max_length', truncation=True,
                                 max_length=self.max_length, return_tensors='pt')
        return {key: val.squeeze(0) for key, val in encoded.items()}


def collate_fn(batch):
    return {
        'input_ids': torch.stack([item['input_ids'] for item in batch]),
        'attention_mask': torch.stack([item['attention_mask'] for item in batch])
    }


def get_bert_representations(model, tokenizer, texts, max_length=128, batch_size=16):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)

    all_cls_representations = []
    all_pooled_representations = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]
            cls_representations = last_hidden_states[:, 0, :]
            pooled_representations = model.pooler(last_hidden_states)

            all_cls_representations.append(cls_representations.cpu())
            all_pooled_representations.append(pooled_representations.cpu())

    cls_representations = torch.cat(all_cls_representations, dim=0)
    pooled_representations = torch.cat(all_pooled_representations, dim=0)

    return cls_representations, pooled_representations


def get_weighted_rep(model, weights, tokenizer, texts, max_length=128, batch_size=16):
    dataset = TextDataset(texts, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                            num_workers=4, pin_memory=True, collate_fn=collate_fn)
    weights = weights.to(model.device)
    out_list = []

    model.eval()
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Processing batches"):
            batch = {k: v.to(model.device) for k, v in batch.items()}

            outputs = model(**batch, output_hidden_states=True)

            last_hidden_states = outputs.hidden_states[-1]
            pooled_representations = model.pooler(last_hidden_states)
            out = pooled_representations * weights
            out_list.append(out.cpu())

    out_list = torch.cat(out_list, dim=0)

    return out_list


def exp(reps, texts):
    src_i = 0
    d = len(reps[0])
    print("num_items", len(reps))
    index = faiss.IndexFlatL2(d)
    index.add(reps)  # add vectors to the index
    scores, indices = index.search(reps[src_i:src_i + 1], k=10)
    for s, i in zip(scores[0], indices[0]):
        print(texts[i], s)


def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


# Example usage:
if __name__ == "__main__":
    subreddit = "TwoXChromosomes"
    model_path = get_model_save_path(f"bert_{subreddit}")  # Adjust this to your model's path
    data = read_csv(get_reddit_train_data_path(subreddit, "train"))
    texts = left(data)
    labels = right(data)
    model = BertModel.from_pretrained(model_path)
    model.to(get_device())

    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.eval()
    cls_reps, pooled_reps = get_bert_representations(model, tokenizer, texts)

    print("CLS Token Representations Shape:", cls_reps.shape)
    print("Pooled Representations Shape:", pooled_reps.shape)
    #
    # # If you want to see the actual values:
    # print("\nCLS Token Representations:")
    # print(cls_reps)
    # print("\nPooled Representations:")
    # print(pooled_reps)
