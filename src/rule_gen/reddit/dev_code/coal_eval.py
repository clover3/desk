import tqdm
import torch
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

from desk_util.path_helper import get_cola_train_data_path, get_model_save_path
from rule_gen.reddit.base_bert.train_bert import load_dataset_from_csv


def evaluate_model(
        model_path,
        test_data_path,
        batch_size=32,
        max_length=256,
        device='cuda' if torch.cuda.is_available() else 'cpu'
):
    # Load the fine-tuned model and tokenizer
    model = BertForSequenceClassification.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    model.to(device)
    model.eval()

    # Load and preprocess the test dataset
    test_dataset = load_dataset_from_csv(test_data_path)

    def tokenize_function(examples):
        return tokenizer(examples['text'], padding='max_length', truncation=True, max_length=max_length)

    tokenized_test = test_dataset.map(tokenize_function, batched=True)
    tokenized_test.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])

    # Create DataLoader
    test_dataloader = DataLoader(tokenized_test, batch_size=batch_size)

    # Evaluation loop
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(test_dataloader, desc="Evaluating"):
            inputs = {k: v.to(device) for k, v in batch.items() if k != 'label'}
            labels = batch['label'].to(device)

            outputs = model(**inputs)
            logits = outputs.logits
            predictions = torch.argmax(logits, dim=-1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate Matthews Correlation Coefficient
    mcc = matthews_corrcoef(all_labels, all_predictions)
    print(f"Matthews Correlation Coefficient: {mcc}")
    return mcc


def main():
    model_name = f"bert_cola"
    evaluate_model(get_model_save_path(model_name),
                   get_cola_train_data_path("val"))


if __name__ == "__main__":
    main()