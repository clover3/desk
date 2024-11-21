import torch
from torch.utils.data import Dataset, DataLoader

from toxicity.cpath import data_root_path
from toxicity.reddit.proto.protory_net_torch import ProtoryNet, train_epoch, evaluate
import os
import pickle
from transformers import BertTokenizer, BertModel


class MyDataset(Dataset):
    def __init__(self, sentences, labels, tokenizer, max_length=128):
        self.sentences = sentences
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        # Join sentences back into a single text for encoding
        text = ' '.join(self.sentences[idx])

        return {
            'text': text,
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


def load_data(dir_path):
    data = {}
    files = ['y_train', 'train_not_clean', 'test_not_clean', 'y_test']

    for file in files:
        file_path = os.path.join(dir_path, file)
        with open(file_path, 'rb') as fp:
            data[file] = pickle.load(fp)
    return data


def gen_sents(para):
    res = []
    for p in para:
        sents = p.split(".")
        # Remove empty strings and strip whitespace
        sents = [s.strip() for s in sents if s.strip()]
        res.append(sents)
    return res


if __name__ == '__main__':
    # Load and prepare data
    dir_path = os.path.join(data_root_path, "hotel")
    data = load_data(dir_path)  # Replace with actual path
    train_not_clean = data['train_not_clean']
    test_not_clean = data['test_not_clean']
    y_train = data['y_train']
    y_test = data['y_test']
    # train_size = 128
    train_size = None
    # Generate sentences
    train_noclean_sents = gen_sents(train_not_clean)
    test_noclean_sents = gen_sents(test_not_clean)
    x_train = train_noclean_sents
    x_test = test_noclean_sents

    # Convert labels to integers
    y_train = [int(y) for y in y_train]
    y_test = [int(y) for y in y_test]

    if train_size:
        x_train = x_train[:train_size]
        y_train = y_train[:train_size]
    # Initialize BERT tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Create datasets
    train_dataset = MyDataset(x_train, y_train, tokenizer)
    test_dataset = MyDataset(x_test, y_test, tokenizer)

    # Create data loaders
    batch_size = 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    eval_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Print dataset information
    print('Training samples:', len(train_noclean_sents))
    print('Test samples:', len(test_noclean_sents))
    print('Sample text:', test_noclean_sents[0])
    print('Training labels sample:', y_train[:10])
    print('Test labels sample:', y_test[:10])

    # Initialize model and move to device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = ProtoryNet(k_protos=10)  # 768 is BERT's hidden size
    model = model.to(device)
    model.init_prototypes(train_loader)
    lr = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Training loop
    num_epochs = 30
    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, optimizer)
        eval_accuracy, eval_loss = evaluate(model, eval_loader)
        print(f"Epoch {epoch}: Train Loss = {train_loss:.4f}, "
              f"Eval Loss = {eval_loss:.4f}, Eval Accuracy = {eval_accuracy:.4f}")

    batch = next(iter(train_loader))
    model.show_prototypes(batch["text"])
